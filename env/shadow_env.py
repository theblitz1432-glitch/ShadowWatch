"""
ShadowWatch-v0 — Core environment logic.
Implements reset(), step(), state(), grade() per OpenEnv spec.
"""

import copy
import math
import random
from typing import Optional

from env.models import (
    Action, Observation, CameraFeed, SensorReadings,
    State, StepResult, ThreatInfo, DroneInfo, ScoreBreakdown,
    THREAT_DESCRIPTIONS,
)
from env.threat_generator import generate_threats, move_threats

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    "single_target_clear": {
        "grid_size": 10, "max_steps": 40, "n_drones": 1,
        "gps_jam_step": None, "gps_spoof": False,
    },
    "multi_threat_gps_denied": {
        "grid_size": 20, "max_steps": 80, "n_drones": 1,
        "gps_jam_step": 20, "gps_spoof": False,
    },
    "swarm_electronic_warfare": {
        "grid_size": 30, "max_steps": 120, "n_drones": 3,
        "gps_jam_step": 10, "gps_spoof": True,
    },
}

CELL_CLEAR  = "clear"
CELL_THREAT = "threat"
CELL_DECOY  = "decoy"
CELL_FOG    = "fog"

WEATHER_OPTIONS = ["clear", "partial_fog", "heavy_fog"]


# ---------------------------------------------------------------------------
# ShadowWatchEnv
# ---------------------------------------------------------------------------

class ShadowWatchEnv:
    """
    OpenEnv-compliant tactical drone patrol environment.

    Interface:
        obs          = env.reset(task_id)
        step_result  = env.step(action_str)
        state        = env.state()
        score        = env.grade()
    """

    def __init__(self, task_id: str = "single_target_clear"):
        self.task_id = task_id
        cfg = TASK_CONFIG[task_id]
        self.grid_size    = cfg["grid_size"]
        self.max_steps    = cfg["max_steps"]
        self.n_drones     = cfg["n_drones"]
        self.gps_jam_step = cfg["gps_jam_step"]
        self.gps_spoof    = cfg["gps_spoof"]

        self.grid:      list[list[str]] = []
        self.threats:   list[dict]      = []
        self.fog_zones: list[dict]      = []
        self.drones:    list[dict]      = []
        self.step_count        = 0
        self.done              = False
        self.gps_status        = "active"
        self.weather           = "clear"
        self.alerts_sent       = 0
        self.threats_confirmed = 0
        self.visited: set      = set()
        self._mobile           = False

        self._exploration_reward  = 0.0
        self._detection_reward    = 0.0
        self._confirmation_reward = 0.0
        self._alert_reward        = 0.0
        self._penalty_total       = 0.0
        self._coordination_bonus  = 0.0

        self.reset(task_id)

    # -----------------------------------------------------------------------
    # reset() — OpenEnv required
    # -----------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset the environment. Returns initial Observation."""
        if task_id and task_id != self.task_id:
            self.__init__(task_id)
            return self._get_observation()

        cfg = TASK_CONFIG[self.task_id]
        self.grid_size    = cfg["grid_size"]
        self.max_steps    = cfg["max_steps"]
        self.n_drones     = cfg["n_drones"]
        self.gps_jam_step = cfg["gps_jam_step"]
        self.gps_spoof    = cfg["gps_spoof"]

        # blank grid
        self.grid = [[CELL_CLEAR] * self.grid_size for _ in range(self.grid_size)]

        # generate threats and fog
        data = generate_threats(self.task_id, self.grid_size)
        self.threats   = data["threats"]
        self.fog_zones = data["fog_zones"]
        self._mobile   = data["mobile"]

        for t in self.threats:
            t["description"] = THREAT_DESCRIPTIONS.get(t["threat_type"], "Unknown threat")
            t["alerted"]     = False

        for t in self.threats:
            r, c = t["position"]
            self.grid[r][c] = CELL_DECOY if t["is_decoy"] else CELL_THREAT

        for fz in self.fog_zones:
            cr, cc, rad = fz["center"][0], fz["center"][1], fz["radius"]
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if math.dist([r, c], [cr, cc]) <= rad:
                        if self.grid[r][c] == CELL_CLEAR:
                            self.grid[r][c] = CELL_FOG

        # place drones
        self.drones = [
            {
                "drone_id":  i + 1,
                "position":  [0, i * max(1, self.grid_size // self.n_drones)],
                "battery":   1.0,
                "is_active": i == 0,
            }
            for i in range(self.n_drones)
        ]

        self.step_count        = 0
        self.done              = False
        self.gps_status        = "active"
        self.weather           = random.choice(WEATHER_OPTIONS)
        self.alerts_sent       = 0
        self.threats_confirmed = 0
        self.visited           = {tuple(self.drones[0]["position"])}

        self._exploration_reward  = 0.0
        self._detection_reward    = 0.0
        self._confirmation_reward = 0.0
        self._alert_reward        = 0.0
        self._penalty_total       = 0.0
        self._coordination_bonus  = 0.0

        return self._get_observation()

    # -----------------------------------------------------------------------
    # step() — OpenEnv required
    # -----------------------------------------------------------------------

    def step(self, action: str) -> StepResult:
        """Execute one action. Returns StepResult(obs, reward, done, info)."""
        if self.done:
            return StepResult(
                obs=self._get_observation(), reward=0.0, done=True,
                info={"message": "Episode already done. Call reset()."}
            )

        Action(action=action)   # raises ValueError if invalid

        reward = 0.0
        info: dict = {}
        active = self._active_drone()

        # GPS jamming event
        if self.gps_jam_step and self.step_count == self.gps_jam_step:
            self.gps_status = "spoofed" if self.gps_spoof else "jammed"
            info["gps_event"] = self.gps_status

        # Weather changes every 15 steps
        if self.step_count % 15 == 0 and self.step_count > 0:
            self.weather = random.choice(WEATHER_OPTIONS)

        # Mobile threats move every 10 steps (Task 3)
        if self._mobile and self.step_count > 0 and self.step_count % 10 == 0:
            self._move_mobile_threats()

        # Execute action
        if action in ("move_north", "move_south", "move_east", "move_west"):
            reward += self._do_move(action, active, info)
        elif action == "hover_scan":
            reward += self._do_hover_scan(active, info)
        elif action == "send_alert":
            reward += self._do_send_alert(active, info)
        elif action == "return_to_base":
            reward += self._do_return_to_base(active, info)
        elif action in ("switch_to_drone_2", "switch_to_drone_3"):
            reward += self._do_switch_drone(action, info)
        elif action == "tag_as_decoy":
            reward += self._do_tag_as_decoy(active, info)

        # Per-step battery drain and penalty
        reward -= 0.01
        self._penalty_total += 0.01
        active["battery"] = max(0.0, active["battery"] - 0.008)

        if active["battery"] == 0.0:
            reward -= 0.50
            self._penalty_total += 0.50
            info["battery_died"] = True
            self.done = True

        self.step_count += 1

        if self.step_count >= self.max_steps:
            self.done = True
            info["reason"] = "max_steps_reached"

        if self.n_drones > 1:
            reward += self._check_coordination()

        return StepResult(
            obs=self._get_observation(),
            reward=round(reward, 4),
            done=self.done,
            info=info,
        )

    # -----------------------------------------------------------------------
    # state() — OpenEnv required
    # -----------------------------------------------------------------------

    def state(self) -> State:
        """Return full ground truth state for grading/debugging."""
        return State(
            full_grid=copy.deepcopy(self.grid),
            all_threats=[ThreatInfo(**t) for t in self.threats],
            all_drones=[DroneInfo(**d) for d in self.drones],
            score_breakdown=ScoreBreakdown(
                exploration_reward=round(self._exploration_reward, 4),
                detection_reward=round(self._detection_reward, 4),
                confirmation_reward=round(self._confirmation_reward, 4),
                alert_reward=round(self._alert_reward, 4),
                penalty_total=round(self._penalty_total, 4),
                coordination_bonus=round(self._coordination_bonus, 4),
                total=round(self._total_reward(), 4),
            ),
            step=self.step_count,
            done=self.done,
            task_id=self.task_id,
        )

    # -----------------------------------------------------------------------
    # grade() — returns 0.0–1.0
    # -----------------------------------------------------------------------

    def grade(self) -> float:
        """Return normalised episode score in [0.0, 1.0]."""
        from env.graders import grade_task1, grade_task2, grade_task3
        s = self.state()
        if self.task_id == "single_target_clear":
            return grade_task1(s)
        elif self.task_id == "multi_threat_gps_denied":
            return grade_task2(s)
        else:
            return grade_task3(s)

    # -----------------------------------------------------------------------
    # Action implementations
    # -----------------------------------------------------------------------

    def _do_move(self, action: str, drone: dict, info: dict) -> float:
        deltas = {
            "move_north": (-1, 0), "move_south": (1, 0),
            "move_east":  (0,  1), "move_west":  (0, -1),
        }
        dr, dc = deltas[action]
        r, c   = drone["position"]
        nr     = max(0, min(self.grid_size - 1, r + dr))
        nc     = max(0, min(self.grid_size - 1, c + dc))
        drone["position"] = [nr, nc]

        if (nr, nc) in self.visited:
            self._penalty_total += 0.05
            return -0.05

        self.visited.add((nr, nc))
        self._exploration_reward += 0.05

        for t in self.threats:
            if t["detected"] and not t["confirmed"]:
                if math.dist([nr, nc], t["position"]) < math.dist([r, c], t["position"]):
                    self._detection_reward += 0.10
                    return 0.15   # explore + chase bonus

        return 0.05

    def _do_hover_scan(self, drone: dict, info: dict) -> float:
        r, c   = drone["position"]
        reward = 0.0
        results = []

        for t in self.threats:
            dist = math.dist([r, c], t["position"])

            if dist <= 2:
                if not t["is_decoy"]:
                    if not t["confirmed"]:
                        t["detected"]  = True
                        t["confirmed"] = True
                        self.threats_confirmed += 1
                        reward += 0.50
                        self._confirmation_reward += 0.50
                        results.append({
                            "threat_type": t["threat_type"],
                            "description": t["description"],
                            "status":      "confirmed",
                            "position":    t["position"],
                        })
                else:
                    reward -= 0.30
                    self._penalty_total += 0.30
                    results.append({
                        "threat_type": t["threat_type"],
                        "status":      "decoy_scanned",
                        "position":    t["position"],
                    })
            elif dist <= 4:
                if not t["detected"]:
                    t["detected"] = True
                    reward += 0.30
                    self._detection_reward += 0.30
                    results.append({
                        "threat_type": t["threat_type"],
                        "description": t["description"],
                        "status":      "detected",
                        "position":    t["position"],
                    })

        info["scan_results"] = results
        return reward

    def _do_send_alert(self, drone: dict, info: dict) -> float:
        """Alert all confirmed, un-alerted real threats. Each threat alerts once."""
        reward  = 0.0
        alerted = []

        for t in self.threats:
            if t["confirmed"] and not t["is_decoy"] and not t.get("alerted", False):
                t["alerted"] = True
                reward += 1.00
                self._alert_reward += 1.00
                self.alerts_sent += 1
                alerted.append({
                    "threat_type": t["threat_type"],
                    "description": t["description"],
                    "position":    t["position"],
                })

        if alerted:
            info["alerts_sent"] = alerted
        else:
            reward -= 0.40
            self._penalty_total += 0.40
            info["false_alarm"] = True

        return reward

    def _do_return_to_base(self, drone: dict, info: dict) -> float:
        drone["position"] = [0, 0]
        drone["battery"]  = 1.0
        info["returned_to_base"] = True
        self._alert_reward += 0.20
        return 0.20

    def _do_switch_drone(self, action: str, info: dict) -> float:
        target = 2 if action == "switch_to_drone_2" else 3
        for d in self.drones:
            d["is_active"] = (d["drone_id"] == target)
        info["switched_to_drone"] = target
        return 0.0

    def _do_tag_as_decoy(self, drone: dict, info: dict) -> float:
        r, c = drone["position"]
        for t in self.threats:
            if t["is_decoy"] and math.dist([r, c], t["position"]) <= 2:
                info["tagged_decoy"] = t["threat_type"]
                return 0.30
        info["wrong_decoy_tag"] = True
        return -0.20

    def _check_coordination(self) -> float:
        positions = [tuple(d["position"]) for d in self.drones]
        if len(set(positions)) < len(positions):
            self._penalty_total += 0.20
            return -0.20
        half = self.grid_size // 2
        quadrants = {(r >= half, c >= half) for r, c in positions}
        if len(quadrants) == len(positions):
            self._coordination_bonus += 0.15
            return 0.15
        return 0.0

    def _move_mobile_threats(self):
        for t in self.threats:
            r, c = t["position"]
            self.grid[r][c] = CELL_CLEAR
        self.threats = move_threats(self.threats, self.grid_size)
        for t in self.threats:
            r, c = t["position"]
            self.grid[r][c] = CELL_DECOY if t["is_decoy"] else CELL_THREAT

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        active = self._active_drone()
        r, c   = active["position"]

        # 5×5 camera window
        local_view = [
            [
                self.grid[r + dr][c + dc]
                if 0 <= r + dr < self.grid_size and 0 <= c + dc < self.grid_size
                else "out_of_bounds"
                for dc in range(-2, 3)
            ]
            for dr in range(-2, 3)
        ]

        # Camera confidence
        confidence = 1.0
        for fz in self.fog_zones:
            if math.dist([r, c], fz["center"]) <= fz["radius"]:
                confidence -= 0.35 if fz["density"] == "heavy_fog" else 0.20
        if self.gps_status in ("jammed", "spoofed"):
            confidence -= 0.10
        confidence = round(max(0.1, confidence), 2)

        # Sensor readings with noise
        mag = thm = mot = 0.0
        for t in self.threats:
            dist = math.dist([r, c], t["position"])
            if dist <= 5:
                signal = max(0.0, 1.0 - dist / 5.0) + random.uniform(-0.05, 0.05)
                if t["threat_type"] == "bunker":
                    mag = min(1.0, mag + signal)
                elif t["threat_type"] in ("military_camp", "convoy"):
                    thm = min(1.0, thm + signal)
                elif t["threat_type"] == "infiltration":
                    mot = min(1.0, mot + signal)

        # GPS spoofing
        if self.gps_status == "spoofed":
            reported_pos = [
                (r + random.randint(-3, 3)) % self.grid_size,
                (c + random.randint(-3, 3)) % self.grid_size,
            ]
        else:
            reported_pos = [r, c]

        return Observation(
            drone_position=reported_pos,
            battery=round(active["battery"], 3),
            step=self.step_count,
            gps_status=self.gps_status,
            camera_feed=CameraFeed(
                local_view_5x5=local_view,
                confidence=confidence,
            ),
            sensor_readings=SensorReadings(
                magnetic=round(max(0.0, min(1.0, mag)), 3),
                thermal=round(max(0.0, min(1.0, thm)), 3),
                motion=round(max(0.0, min(1.0, mot)), 3),
            ),
            weather=self.weather,
            alerts_sent=self.alerts_sent,
            threats_confirmed=self.threats_confirmed,
        )

    def _active_drone(self) -> dict:
        for d in self.drones:
            if d["is_active"]:
                return d
        return self.drones[0]

    def _total_reward(self) -> float:
        return (
            self._exploration_reward
            + self._detection_reward
            + self._confirmation_reward
            + self._alert_reward
            + self._coordination_bonus
            - self._penalty_total
        )