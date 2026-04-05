"""
ShadowWatch-v0 — Typed Pydantic models
Implements the full OpenEnv observation/action/state interface.
"""

from pydantic import BaseModel, field_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "hover_scan",
    "send_alert",
    "return_to_base",
    "switch_to_drone_2",
    "switch_to_drone_3",
    "tag_as_decoy",
]

THREAT_DESCRIPTIONS = {
    "bunker":        "Underground fortification — magnetic anomaly present",
    "military_camp": "Military encampment — thermal signatures and vehicle activity",
    "convoy":        "Armed convoy in restricted zone — multiple heat signatures",
    "infiltration":  "Unauthorised personnel crossing perimeter — motion detected",
}


class Action(BaseModel):
    """Validated action model. Raises ValueError on invalid action string."""
    action: str

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: str) -> str:
        if v not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{v}'. Must be one of: {VALID_ACTIONS}"
            )
        return v


# ---------------------------------------------------------------------------
# Observation model — what the agent sees each step
# ---------------------------------------------------------------------------

class CameraFeed(BaseModel):
    """5×5 local camera view centred on the drone."""
    local_view_5x5: list[list[str]]   # cell labels: clear/threat/decoy/fog/out_of_bounds
    confidence: float                  # 0.0–1.0, reduced by fog and jamming


class SensorReadings(BaseModel):
    """
    Passive sensor readings at the drone's current position.
    Each channel reflects a specific threat category:
      magnetic → bunker
      thermal  → military_camp or convoy
      motion   → infiltration
    """
    magnetic: float    # 0.0–1.0
    thermal:  float    # 0.0–1.0
    motion:   float    # 0.0–1.0


class Observation(BaseModel):
    """
    Full observation returned by reset() and step().
    GPS position may be incorrect when gps_status is 'spoofed'.
    """
    drone_position:    list[int]       # [row, col] — reported GPS coords
    battery:           float           # 0.0–1.0
    step:              int             # current step number
    gps_status:        str             # "active" | "jammed" | "spoofed"
    camera_feed:       CameraFeed
    sensor_readings:   SensorReadings
    weather:           str             # "clear" | "partial_fog" | "heavy_fog"
    alerts_sent:       int             # cumulative alerts sent this episode
    threats_confirmed: int             # cumulative threats confirmed this episode


# ---------------------------------------------------------------------------
# State model — full ground truth (judges / debugging only)
# ---------------------------------------------------------------------------

class ThreatInfo(BaseModel):
    """Ground truth information about a single threat or decoy."""
    threat_type:  str              # "bunker" | "military_camp" | "convoy" | "infiltration"
    position:     list[int]        # [row, col] true position
    is_decoy:     bool
    detected:     bool = False     # drone came within detection range
    confirmed:    bool = False     # drone confirmed with hover_scan
    alerted:      bool = False     # send_alert was called for this threat
    description:  str  = ""        # human-readable description

    def __init__(self, **data):
        if not data.get("description"):
            data["description"] = THREAT_DESCRIPTIONS.get(
                data.get("threat_type", ""), "Unknown threat"
            )
        super().__init__(**data)


class DroneInfo(BaseModel):
    """State of a single drone."""
    drone_id:  int
    position:  list[int]
    battery:   float
    is_active: bool


class ScoreBreakdown(BaseModel):
    """Detailed reward accounting for the current episode."""
    exploration_reward:  float
    detection_reward:    float
    confirmation_reward: float
    alert_reward:        float
    penalty_total:       float
    coordination_bonus:  float
    total:               float


class State(BaseModel):
    """
    Full ground truth state — returned by state() endpoint.
    Agents do not receive this; it is used for grading and debugging.
    """
    full_grid:       list[list[str]]   # complete grid with all cell labels
    all_threats:     list[ThreatInfo]
    all_drones:      list[DroneInfo]
    score_breakdown: ScoreBreakdown
    step:            int
    done:            bool
    task_id:         str


# ---------------------------------------------------------------------------
# StepResult model — returned by step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Return type of env.step(). Contains next observation, reward, and done flag."""
    obs:    Observation
    reward: float
    done:   bool
    info:   dict