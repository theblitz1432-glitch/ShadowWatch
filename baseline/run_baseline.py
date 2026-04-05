"""
ShadowWatch-v0 — Baseline Agent
Runs all 3 tasks and prints reproducible baseline scores.
Uses OpenAI-compatible client — works with Gemini, HuggingFace, or any OpenAI-compatible API.
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL     = os.getenv("SHADOWWATCH_API_URL", "http://localhost:7860")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gemini-2.0-flash")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASKS = [
    "single_target_clear",
    "multi_threat_gps_denied",
    "swarm_electronic_warfare",
]

SYSTEM_PROMPT = """You are a tactical drone patrol agent for the ShadowWatch defence system.
Detect and report: bunkers (magnetic sensor), military camps/convoys (thermal), infiltrators (motion).

Rules:
- Explore new cells (+0.05), move toward signals (+0.10)
- hover_scan when sensor > 0.7 to confirm threats (+0.50)
- send_alert ONLY after confirming (+1.00 correct / -0.40 false alarm)
- return_to_base if battery < 0.25
- Avoid revisiting cells (-0.05)

Actions: move_north, move_south, move_east, move_west,
         hover_scan, send_alert, return_to_base,
         switch_to_drone_2, switch_to_drone_3, tag_as_decoy

Respond with ONLY the action string."""


def choose_action(obs: dict) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Observation:\n{json.dumps(obs, indent=2)}\n\nChoose action:"},
            ],
            max_tokens=20,
            temperature=0.2,
        )
        action = (response.choices[0].message.content or "").strip().lower()
        return action.strip(".,!?\"'")
    except Exception:
        return "move_north"


def run_episode(task_id: str) -> float:
    print(f"\n{'='*50}\nTask: {task_id}\n{'='*50}")

    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    obs  = r.json()
    step = 0
    done = False

    while not done:
        action  = choose_action(obs)
        sensors = obs.get("sensor_readings", {})
        print(
            f"  Step {step:03d} | {action:<22} | "
            f"battery={obs.get('battery', 0):.2f} | "
            f"mag={sensors.get('magnetic',0):.2f} "
            f"thm={sensors.get('thermal',0):.2f} "
            f"mot={sensors.get('motion',0):.2f} | "
            f"confirmed={obs.get('threats_confirmed',0)} "
            f"alerted={obs.get('alerts_sent',0)}"
        )

        r = requests.post(f"{BASE_URL}/step", json={"action": action})
        if r.status_code == 400:
            print(f"  ⚠ Bad action '{action}', falling back to move_north")
            r = requests.post(f"{BASE_URL}/step", json={"action": "move_north"})

        r.raise_for_status()
        result = r.json()
        obs    = result["obs"]
        done   = result["done"]
        step  += 1

    r     = requests.get(f"{BASE_URL}/grade")
    score = r.json()["score"]
    print(f"\n  ✅ Final score: {score:.4f}  (steps: {step})")
    return score


def main():
    print("ShadowWatch-v0 — Baseline Agent")
    print("=" * 50)

    scores = {}
    for task in TASKS:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"  ❌ {task} failed: {e}")
            scores[task] = 0.0

    print("\n" + "=" * 50)
    print("BASELINE SCORES SUMMARY")
    print("=" * 50)
    print(f"{'Task':<35} {'Score':>8}")
    print("-" * 45)
    for task, score in scores.items():
        print(f"{task:<35} {score:>8.4f}")
    print("-" * 45)
    avg = sum(scores.values()) / len(scores)
    print(f"{'Average':<35} {avg:>8.4f}")
    print("\nCopy these scores into your README.md baseline table.")


if __name__ == "__main__":
    main()