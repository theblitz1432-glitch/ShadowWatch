"""
ShadowWatch-v0  —  Inference Script (v6 - Scan Every Cell)
===========================================================
Strategy:
  - Snake walk covering ALL cells
  - hover_scan at EVERY cell (guaranteed to find threat anywhere)
  - send_alert immediately on confirm
  - return_to_base after alert (stops penalty bleed)
  - Break loop the moment env says done=True
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
TASK_NAME       = os.getenv("SHADOWWATCH_TASK", "single_target_clear")
BENCHMARK       = "shadowwatch-v0"

MAX_STEPS         = 50
SUCCESS_THRESHOLD = 0.4
GRID_SIZE         = 10

MOVE_ACTIONS = ["move_north", "move_south", "move_east", "move_west"]


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END]   success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def get_max_signal(obs: dict) -> float:
    s = obs.get("sensor_readings", {})
    if not isinstance(s, dict): return 0.0
    return max(s.get("magnetic", 0.0), s.get("thermal", 0.0), s.get("motion", 0.0))


# ── Interleaved scan+move path ────────────────────────────────────────────────
# For each cell in snake order: move to it, then hover_scan it.
# This guarantees every cell is scanned within range.
def build_scan_plan(grid_size: int) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Returns list of (action_type, target_cell):
      ('move', (r,c))  → move toward this cell
      ('scan', (r,c))  → hover_scan at this cell
    """
    plan = []
    snake = []
    for r in range(grid_size):
        cols = range(grid_size) if r % 2 == 0 else range(grid_size - 1, -1, -1)
        for c in cols:
            snake.append((r, c))
    for cell in snake:
        plan.append(('move', cell))
        plan.append(('scan', cell))
    return plan


def move_toward(current: Tuple[int,int], target: Tuple[int,int]) -> str:
    cr, cc = current
    tr, tc = target
    if abs(tr - cr) >= abs(tc - cc):
        if tr < cr: return "move_north"
        if tr > cr: return "move_south"
    if tc > cc: return "move_east"
    if tc < cc: return "move_west"
    return random.choice(MOVE_ACTIONS)


def print_ground_truth(url: str):
    try:
        resp = requests.get(f"{url}/state", timeout=30)
        if resp.status_code != 200: return
        state = resp.json()
        print("\n========== GROUND TRUTH ==========", flush=True)
        for t in state.get("all_threats", []):
            decoy = " (DECOY)" if t.get("is_decoy") else ""
            print(f"[TRUTH] {t['threat_type'].upper()}{decoy} at {t['position']}"
                  f" | detected={t.get('detected')} confirmed={t.get('confirmed')} alerted={t.get('alerted')}", flush=True)
        print(f"[TRUTH] Breakdown: {state.get('score_breakdown')}", flush=True)
        print("===================================\n", flush=True)
    except Exception as e:
        print(f"[TRUTH] {e}", flush=True)


def step_env(url: str, action: str):
    resp = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    if resp.status_code == 400:
        # Invalid action fallback
        action = random.choice(MOVE_ACTIONS)
        resp = requests.post(f"{url}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json(), action


def run_episode(task_id: str, client: OpenAI) -> float:
    rewards:      List[float] = []
    steps_taken:  int         = 0
    success:      bool        = False
    score:        float       = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(f"{SHADOWWATCH_URL}/reset",
                             json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs  = resp.json()
        done = False

        pos           = tuple(obs.get("drone_position", [0, 0]))
        mission_done  = False   # True after alert sent
        plan          = build_scan_plan(GRID_SIZE)
        plan_idx      = 0
        scanned_cells: Set[Tuple] = set()

        step = 0
        while step < MAX_STEPS and not done:
            step += 1

            pos               = tuple(obs.get("drone_position", [0, 0]))
            battery           = obs.get("battery", 1.0)
            threats_confirmed = obs.get("threats_confirmed", 0)
            alerts_sent       = obs.get("alerts_sent", 0)
            max_sig           = get_max_signal(obs)

            print(f"[DBG]   step={step} pos={pos} sig={max_sig:.3f} "
                  f"bat={battery:.2f} conf={threats_confirmed} alerted={alerts_sent}", flush=True)

            # ── Priority pipeline ──────────────────────────────────────────

            # P1: Battery critical → recharge
            if battery < 0.18:
                action = "return_to_base"
                print("[DBG]   → BATTERY LOW → return_to_base", flush=True)

            # P2: Alert done → return to base (but only once)
            elif mission_done:
                action = "return_to_base"
                print("[DBG]   → MISSION DONE → return_to_base", flush=True)

            # P3: Confirmed but not alerted
            elif threats_confirmed > alerts_sent:
                action = "send_alert"
                print("[DBG]   → SEND ALERT", flush=True)

            # P4: Very strong signal → scan immediately (skip plan)
            elif max_sig > 0.55 and pos not in scanned_cells:
                action = "hover_scan"
                scanned_cells.add(pos)
                print(f"[DBG]   → OPPORTUNISTIC SCAN sig={max_sig:.3f}", flush=True)

            # P5: Follow scan plan
            else:
                # Advance plan past already-done items
                while plan_idx < len(plan):
                    ptype, ptarget = plan[plan_idx]
                    if ptype == 'move' and ptarget == pos:
                        plan_idx += 1  # already here
                    elif ptype == 'scan' and ptarget in scanned_cells:
                        plan_idx += 1  # already scanned
                    else:
                        break

                if plan_idx >= len(plan):
                    action = "return_to_base"
                    print("[DBG]   → PLAN COMPLETE → return_to_base", flush=True)
                else:
                    ptype, ptarget = plan[plan_idx]
                    if ptype == 'move':
                        if ptarget == pos:
                            plan_idx += 1
                            action = "hover_scan"  # scan current cell
                            scanned_cells.add(pos)
                            print(f"[DBG]   → AT TARGET → hover_scan", flush=True)
                        else:
                            action = move_toward(pos, ptarget)
                            print(f"[DBG]   → MOVE to {ptarget} → {action}", flush=True)
                    else:  # scan
                        if ptarget == pos:
                            action = "hover_scan"
                            scanned_cells.add(pos)
                            plan_idx += 1
                            print(f"[DBG]   → SCAN at {pos}", flush=True)
                        else:
                            action = move_toward(pos, ptarget)
                            print(f"[DBG]   → MOVE to scan target {ptarget} → {action}", flush=True)

            # ── Execute ────────────────────────────────────────────────────
            result, action = step_env(SHADOWWATCH_URL, action)
            obs    = result["obs"]
            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))

            info = result.get("info", {})
            if info.get("scan_results"):
                for sr in info["scan_results"]:
                    print(f"[SCAN]  {sr['status'].upper()} {sr['threat_type']} at {sr['position']}", flush=True)
            if info.get("alerts_sent"):
                for alert in info["alerts_sent"]:
                    print(f"[ALERT] ✓ {alert['threat_type'].upper()} at {alert['position']}", flush=True)
                mission_done = True
            if info.get("false_alarm"):
                print("[WARN]  False alarm!", flush=True)
            if info.get("returned_to_base"):
                print("[INFO]  Base reached — battery recharged.", flush=True)
                if mission_done:
                    print("[INFO]  Mission complete. Stopping.", flush=True)
                    rewards.append(reward)
                    steps_taken = step
                    log_step(step=step, action=action, reward=reward, done=done)
                    break   # ← exit cleanly after recharge post-alert

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done)

            if done:
                break   # ← always break on env done

        # ── Grade ──────────────────────────────────────────────────────────
        resp = requests.get(f"{SHADOWWATCH_URL}/grade", timeout=30)
        resp.raise_for_status()
        score   = float(resp.json().get("score", 0.0))
        success = score >= SUCCESS_THRESHOLD
        print_ground_truth(SHADOWWATCH_URL)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    run_episode(task_id=TASK_NAME, client=client)

if __name__ == "__main__":
    main()