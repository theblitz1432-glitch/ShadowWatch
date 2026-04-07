"""
ShadowWatch-v0  —  Inference Script (v10 - Scan-first + decoy cooldown)
========================================================================
Fixes over v9:
  1. Signal > 0.6 triggers hover_scan BEFORE checking camera direction —
     previously camera always fired first, causing infinite movement loops
     even when the drone was sitting on top of the target.
  2. Decoy cooldown: after DECOY_SCANNED the agent waits DECOY_COOLDOWN
     steps before scanning again, stopping the -0.16 × 30 penalty spiral.
  3. Reset chasing/best_signal properly after a confirmed scan so the
     drone moves on to find the next threat.
"""

import os
import random
from typing import List, Set, Tuple, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY         = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL    = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME      = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
SHADOWWATCH_URL = os.getenv("SHADOWWATCH_API_URL", "http://localhost:7860")
BENCHMARK       = "shadowwatch-v0"

SUCCESS_THRESHOLD = 0.4
LOW_BATTERY       = 0.15
DECOY_COOLDOWN    = 15   # steps to skip scanning after a decoy result

TASK_CONFIG = {
    "single_target_clear":      {"grid": 10, "max_steps": 40,  "drones": 1},
    "multi_threat_gps_denied":  {"grid": 20, "max_steps": 80,  "drones": 1},
    "swarm_electronic_warfare": {"grid": 30, "max_steps": 120, "drones": 3},
}

MOVE_ACTIONS = ["move_north", "move_south", "move_east", "move_west"]


# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END]   success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ── Observation helpers ────────────────────────────────────────────────────────

def get_signals(obs):
    s = obs.get("sensor_readings", {})
    if not isinstance(s, dict): return 0.0, 0.0, 0.0
    return s.get("magnetic", 0.0), s.get("thermal", 0.0), s.get("motion", 0.0)

def get_max_signal(obs): return max(get_signals(obs))

def build_snake(grid_size):
    cells = []
    for r in range(grid_size):
        cols = range(grid_size) if r % 2 == 0 else range(grid_size - 1, -1, -1)
        for c in cols: cells.append((r, c))
    return cells

def move_toward(cur, tgt):
    cr, cc = cur; tr, tc = tgt
    if abs(tr - cr) >= abs(tc - cc):
        if tr < cr: return "move_north"
        if tr > cr: return "move_south"
    if tc > cc: return "move_east"
    if tc < cc: return "move_west"
    return random.choice(MOVE_ACTIONS)

def next_pos(pos, move, gs):
    r, c = pos
    if move == "move_north": return (max(0, r - 1), c)
    if move == "move_south": return (min(gs - 1, r + 1), c)
    if move == "move_east":  return (r, min(gs - 1, c + 1))
    if move == "move_west":  return (r, max(0, c - 1))
    return pos

def battery_reserve(pos):
    return (pos[0] + pos[1]) * 0.009 + LOW_BATTERY

def camera_direction(obs):
    """Find 'threat' cell in 5×5 camera feed and return move direction."""
    feed = obs.get("camera_feed", {})
    grid = feed.get("local_view_5x5", []) if isinstance(feed, dict) else []
    if not grid: return None
    best_dist = 99; best_dir = None
    for dr in range(5):
        for dc in range(5):
            if grid[dr][dc] == "threat":
                ro, co = dr - 2, dc - 2
                dist = abs(ro) + abs(co)
                if dist < best_dist:
                    best_dist = dist
                    if abs(ro) >= abs(co):
                        best_dir = "move_north" if ro < 0 else "move_south"
                    else:
                        best_dir = "move_east" if co > 0 else "move_west"
    return best_dir

def step_env(url, action):
    resp = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    if resp.status_code == 400:
        action = random.choice(MOVE_ACTIONS)
        resp = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json(), action

def print_ground_truth(url):
    try:
        resp = requests.get(f"{url}/state", timeout=30)
        if resp.status_code != 200: return
        state = resp.json()
        print("\n========== GROUND TRUTH ==========", flush=True)
        for t in state.get("all_threats", []):
            d = " (DECOY)" if t.get("is_decoy") else ""
            print(f"[TRUTH] {t['threat_type'].upper()}{d} at {t['position']}"
                  f" | det={t.get('detected')} conf={t.get('confirmed')} alerted={t.get('alerted')}", flush=True)
        print(f"[TRUTH] {state.get('score_breakdown')}", flush=True)
        print("===================================\n", flush=True)
    except Exception as e:
        print(f"[TRUTH] {e}", flush=True)


# ── Main episode runner ────────────────────────────────────────────────────────

def run_episode(task_id: str, client) -> float:
    cfg       = TASK_CONFIG[task_id]
    grid_size = cfg["grid"]
    max_steps = cfg["max_steps"]
    n_drones  = cfg["drones"]

    rewards: List[float] = []
    steps_taken = 0; success = False; score = 0.0

    snake     = build_snake(grid_size)
    snake_idx = 0
    visited:  Set[Tuple] = set()
    scanned:  Set[Tuple] = set()

    # Chase state
    chasing     = False
    prev_signal = 0.0
    best_signal = 0.0
    best_pos: Optional[Tuple] = None
    last_move   = "move_south"

    # Drone rotation (task 3)
    active_drone = 1

    # ── FIX 2: decoy cooldown ──────────────────────────────────────────────
    decoy_cooldown_remaining = 0   # steps left where scanning is suppressed

    # ── FIX 3: post-alert cooldown ─────────────────────────────────────────
    # After sending an alert the drone is still near the confirmed target, so
    # signal stays high and chasing re-enables immediately, causing the N/S
    # oscillation loop. Force snake-walk for POST_ALERT_COOLDOWN steps after
    # each alert so the drone moves away before hunting the next signal.
    POST_ALERT_COOLDOWN      = 8
    post_alert_cooldown_rem  = 0
    CLEARED_RADIUS           = 4   # cells around a confirmed+alerted threat
    cleared_zones: list      = []  # list of (row, col) where alerts were sent

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(f"{SHADOWWATCH_URL}/reset",
                             json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json(); done = False

        for step in range(1, max_steps + 1):
            if done: break

            pos               = tuple(obs.get("drone_position", [0, 0]))
            battery           = obs.get("battery", 1.0)
            threats_confirmed = obs.get("threats_confirmed", 0)
            alerts_sent       = obs.get("alerts_sent", 0)
            gps               = obs.get("gps_status", "active")
            max_sig           = get_max_signal(obs)
            reserve           = battery_reserve(pos)

            visited.add(pos)

            # Countdown timers
            if decoy_cooldown_remaining > 0:
                decoy_cooldown_remaining -= 1
            if post_alert_cooldown_rem > 0:
                post_alert_cooldown_rem -= 1

            # Re-enable chasing only when post-alert cooldown has cleared,
            # preventing immediate re-lock on the just-confirmed target.
            if max_sig > best_signal:
                best_signal = max_sig; best_pos = pos
            in_cleared = any(
                abs(pos[0]-c[0]) + abs(pos[1]-c[1]) <= CLEARED_RADIUS
                for c in cleared_zones
            )
            if max_sig > 0.1 and not chasing and post_alert_cooldown_rem == 0 and not in_cleared:
                chasing = True

            # Convenience flag: may we scan this step?
            scan_allowed = (decoy_cooldown_remaining == 0) and (pos not in scanned)

            print(f"[DBG]   step={step} d={active_drone} pos={pos} "
                  f"sig={max_sig:.3f} bat={battery:.2f} gps={gps} "
                  f"conf={threats_confirmed} alerted={alerts_sent} chase={chasing}", flush=True)

            # ── Decision pipeline ──────────────────────────────────────────

            # P1: Low battery → recharge
            if battery <= reserve:
                action = "return_to_base"
                print(f"[DBG]   → LOW BAT → return_to_base", flush=True)

            # P2: New confirmed threat → alert
            elif threats_confirmed > alerts_sent:
                action = "send_alert"
                print(f"[DBG]   → ALERT", flush=True)

            # P3: Drone rotation for task 3 (every 35 steps)
            elif n_drones > 1 and step % 35 == 0:
                nxt = (active_drone % n_drones) + 1
                action = f"switch_to_drone_{nxt}"
                active_drone = nxt
                print(f"[DBG]   → SWITCH → drone {nxt}", flush=True)

            # P4: GPS spoofed — position unreliable, use signal only
            elif gps == "spoofed":
                # FIX 1+2: scan first if signal strong AND cooldown clear
                if max_sig > 0.6 and scan_allowed:
                    action = "hover_scan"
                    scanned.add(pos)
                    print(f"[DBG]   → SPOOFED+STRONG → scan", flush=True)
                elif max_sig > prev_signal:
                    action = last_move
                    print(f"[DBG]   → SPOOFED+RISING → {action}", flush=True)
                else:
                    action = random.choice(["move_south", "move_east", "move_south", "move_west"])
                    print(f"[DBG]   → SPOOFED+PROBE → {action}", flush=True)

            # P5: Chasing a signal (normal GPS or jammed)
            elif chasing and max_sig > 0.1:
                # Abort chase if we've drifted back into an already-cleared zone
                if any(abs(pos[0]-c[0]) + abs(pos[1]-c[1]) <= CLEARED_RADIUS for c in cleared_zones):
                    chasing = False
                    best_signal = 0.0; best_pos = None
                # FIX 1: scan BEFORE checking camera when signal is strong
                if max_sig > 0.6 and scan_allowed:
                    action = "hover_scan"
                    scanned.add(pos)
                    print(f"[DBG]   → CHASE+CONFIRM sig={max_sig:.3f} → scan", flush=True)
                else:
                    cam = camera_direction(obs)
                    if cam:
                        action = cam
                        print(f"[DBG]   → CHASE+CAMERA → {action}", flush=True)
                    elif max_sig >= prev_signal:
                        # Signal rising or stable — keep heading same direction
                        cands = [m for m in MOVE_ACTIONS
                                 if next_pos(pos, m, grid_size) not in visited]
                        action = cands[0] if cands else last_move
                        print(f"[DBG]   → CHASE+RISING → {action}", flush=True)
                    elif best_pos and best_pos != pos:
                        action = move_toward(pos, best_pos)
                        print(f"[DBG]   → CHASE+BACKTRACK to {best_pos} → {action}", flush=True)
                    else:
                        chasing = False
                        action = random.choice(MOVE_ACTIONS)

            # P6: GPS jammed — trust signal, continue snake
            elif gps == "jammed":
                if max_sig > 0.6 and scan_allowed:
                    action = "hover_scan"
                    scanned.add(pos)
                    print(f"[DBG]   → JAMMED+SCAN sig={max_sig:.3f}", flush=True)
                elif chasing and max_sig > 0.1:
                    cam = camera_direction(obs)
                    if cam and next_pos(pos, cam, grid_size) not in visited:
                        action = cam
                    else:
                        cands = [m for m in MOVE_ACTIONS
                                 if next_pos(pos, m, grid_size) not in visited]
                        action = cands[0] if cands else last_move
                    print(f"[DBG]   → JAMMED+CHASE → {action}", flush=True)
                else:
                    cands = [m for m in ["move_south", "move_east", "move_north", "move_west"]
                             if next_pos(pos, m, grid_size) not in visited]
                    action = cands[0] if cands else "move_south"
                    print(f"[DBG]   → JAMMED+MOVE → {action}", flush=True)

            # P7: Normal snake — move only, scan only on strong signal
            else:
                while snake_idx < len(snake) and snake[snake_idx] in visited:
                    snake_idx += 1

                if snake_idx >= len(snake):
                    action = "return_to_base"
                    print(f"[DBG]   → DONE → return_to_base", flush=True)
                else:
                    target = snake[snake_idx]
                    if pos == target:
                        snake_idx += 1
                        if max_sig > 0.3 and scan_allowed:
                            action = "hover_scan"
                            scanned.add(pos)
                            print(f"[DBG]   → AT TARGET+SIGNAL → scan", flush=True)
                        else:
                            while snake_idx < len(snake) and snake[snake_idx] in visited:
                                snake_idx += 1
                            if snake_idx < len(snake):
                                action = move_toward(pos, snake[snake_idx])
                                print(f"[DBG]   → SNAKE → {action}", flush=True)
                            else:
                                action = "return_to_base"
                    else:
                        action = move_toward(pos, target)
                        print(f"[DBG]   → SNAKE to {target} → {action}", flush=True)

            prev_signal = max_sig
            if action in MOVE_ACTIONS: last_move = action

            result, action = step_env(SHADOWWATCH_URL, action)
            obs    = result["obs"]
            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))

            info = result.get("info", {})
            if info.get("gps_event"):
                print(f"[GPS]   ⚠ {info['gps_event']}", flush=True)

            if info.get("scan_results"):
                for sr in info["scan_results"]:
                    status = sr['status'].upper()
                    print(f"[SCAN]  {status} {sr['threat_type']} at {sr['position']}", flush=True)
                    # FIX 2: decoy cooldown — stop scanning for a while
                    if "DECOY" in status:
                        decoy_cooldown_remaining = DECOY_COOLDOWN
                        # Also reset chasing so we move away and search elsewhere
                        chasing = False; best_signal = 0.0; best_pos = None
                        print(f"[DBG]   Decoy detected — scan cooldown {DECOY_COOLDOWN} steps, resuming search", flush=True)
                    elif "CONFIRMED" in status or "DETECTED" in status:
                        # Real threat found — reset so we hunt the next one
                        chasing = False; best_signal = 0.0; best_pos = None

            if info.get("alerts_sent"):
                for a in info["alerts_sent"]:
                    print(f"[ALERT] ✓ {a['threat_type'].upper()} at {a['position']}", flush=True)
                # FIX 3+4: record cleared zone AND start cooldown
                cleared_zones.append(pos)
                chasing = False; best_signal = 0.0; best_pos = None
                post_alert_cooldown_rem = POST_ALERT_COOLDOWN

            if info.get("false_alarm"):
                print("[WARN]  False alarm!", flush=True)

            if info.get("returned_to_base"):
                new_bat = obs.get("battery", 1.0)
                print(f"[INFO]  Base. bat→{new_bat:.2f}", flush=True)
                chasing = False; best_signal = 0.0; best_pos = None
                post_alert_cooldown_rem = 0  # base reset clears cooldown too

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done)
            if done: break

        resp = requests.get(f"{SHADOWWATCH_URL}/grade", timeout=30)
        resp.raise_for_status()
        score   = float(resp.json().get("score", 0.0))
        success = score >= SUCCESS_THRESHOLD
        print_ground_truth(SHADOWWATCH_URL)

    except Exception as exc:
        print(f"[DEBUG] {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks  = ["single_target_clear", "multi_threat_gps_denied", "swarm_electronic_warfare"]
    scores = {}

    for task in tasks:
        print(f"\n{'='*60}\nRUNNING: {task}\n{'='*60}\n", flush=True)
        scores[task] = run_episode(task_id=task, client=client)

    print(f"\n{'='*60}\nFINAL SCORES:\n{'='*60}", flush=True)
    for task, s in scores.items():
        status = "✓ PASS" if s >= SUCCESS_THRESHOLD else "✗ FAIL"
        print(f"  {status}  {task}: {s:.3f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()