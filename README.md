---
title: ShadowWatch
emoji: 👁️
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
---
# ShadowWatch-v0

**Autonomous Drone Patrol & Threat Detection — OpenEnv Environment**

---

## Motivation

ShadowWatch-v0 simulates a real-world challenge in autonomous defence: an unmanned aerial vehicle (UAV) must patrol a grid, detect genuine threats using noisy multi-sensor data, and send mission-critical alerts — while managing limited battery, avoiding decoys planted by adversaries, and operating under GPS jamming and electronic warfare.

This environment is directly useful for training and evaluating agents on:
- Sensor fusion under uncertainty
- Sparse reward exploration (large grids, few threats)
- Adversarial deception (decoys, spoofed GPS)
- Multi-agent coordination (Task 3)

---

## Tasks

| Task ID | Difficulty | Grid | Drones | Description |
|---------|-----------|------|--------|-------------|
| `single_target_clear` | Easy | 10×10 | 1 | Locate and confirm one infiltrator. Clear GPS, no fog. |
| `multi_threat_gps_denied` | Medium | 20×20 | 1 | Find 3 threats (bunker, camp, convoy). GPS jams at step 20. Fog zones present. |
| `swarm_electronic_warfare` | Hard | 30×30 | 3 | 3 real threats + 2 decoys. GPS spoofed from step 10. Mobile threats. 3-drone coordination required. |

---

## Threat Types

| Type | Sensor | Description |
|------|--------|-------------|
| `bunker` | `magnetic` | Underground fortification — magnetic anomaly present |
| `military_camp` | `thermal` | Encampment with heat signatures and vehicle activity |
| `convoy` | `thermal` | Armed vehicles moving through restricted zone |
| `infiltration` | `motion` | Unauthorised personnel crossing the perimeter |

---

## Action Space

| Action | Description | Reward |
|--------|-------------|--------|
| `move_north/south/east/west` | Move drone one cell | +0.05 new cell / −0.05 revisit |
| `hover_scan` | Confirm threats within radius 2 | +0.50 confirm / +0.30 detect |
| `send_alert` | Report confirmed threat | +1.00 correct / −0.40 false alarm |
| `return_to_base` | Fly to [0,0] and recharge | +0.20 |
| `switch_to_drone_2/3` | Activate another drone (Task 3) | 0.00 |
| `tag_as_decoy` | Mark cell as decoy | +0.30 correct / −0.20 wrong |

---

## Observation Space

```json
{
  "drone_position":    [row, col],
  "battery":           0.0–1.0,
  "step":              int,
  "gps_status":        "active | jammed | spoofed",
  "camera_feed": {
    "local_view_5x5":  [["clear","threat","fog",...], ...],
    "confidence":      0.0–1.0
  },
  "sensor_readings": {
    "magnetic": 0.0–1.0,
    "thermal":  0.0–1.0,
    "motion":   0.0–1.0
  },
  "weather":           "clear | partial_fog | heavy_fog",
  "alerts_sent":       int,
  "threats_confirmed": int
}
```

---

## Reward Function

The reward is shaped across the full episode (not just terminal):

| Signal | Reward |
|--------|--------|
| Explore new cell | +0.05 |
| Move toward detected threat | +0.10 |
| Detect threat (range 2–4) | +0.30 |
| Confirm threat (hover_scan) | +0.50 |
| Correct alert | +1.00 |
| Return to base | +0.20 |
| Correct decoy tag | +0.30 |
| Revisit penalty | −0.05 |
| False alarm | −0.40 |
| Scan decoy | −0.30 |
| Wrong decoy tag | −0.20 |
| Per-step battery cost | −0.01 |
| Battery dies | −0.50 |
| Drones overlap | −0.20 |
| Multi-drone quadrant bonus | +0.15 |

---

## Grading (0.0–1.0)

**Task 1:** `detection_accuracy × speed_bonus × battery_remaining`

**Task 2:** `threats_found × location_accuracy × (1 − false_alarm_rate)`

**Task 3:** `detection_recall × decoy_rejection × report_quality × coordination`

---

## Baseline Scores

| Task | Score |
|------|-------|
| single_target_clear | 0.0000 |
| multi_threat_gps_denied | 0.0000 |
| swarm_electronic_warfare | 0.0000 |
| **Average** | **0.0000** |

> Run `python baseline/run_baseline.py` to reproduce. Requires API credentials in `.env`.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — add your HF_TOKEN

# 3. Start the server (Terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 7860

# 4. Run inference (Terminal 2)
python inference.py
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace / API key |
| `API_BASE_URL` | LLM endpoint (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `SHADOWWATCH_API_URL` | Server URL (default: `http://localhost:7860`) |
| `SHADOWWATCH_TASK` | Task to run (default: `single_target_clear`) |

---

## Docker

```bash
docker build -t shadowwatch .
docker run -p 7860:7860 shadowwatch
```

---

## Project Structure

```
ShadowWatch/
├── api/
│   └── main.py              # FastAPI server (reset/step/state/grade)
├── baseline/
│   └── run_baseline.py      # Baseline agent
├── env/
│   ├── graders.py           # Task graders — deterministic, 0.0–1.0
│   ├── models.py            # Typed Pydantic models (OpenEnv spec)
│   ├── shadow_env.py        # Core environment: reset/step/state/grade
│   └── threat_generator.py  # Threat placement and fog generation
├── inference.py             # Submission script (strict stdout format)
├── openenv.yaml             # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset env, returns `Observation` |
| `/step` | POST | Execute action, returns `StepResult` |
| `/state` | GET | Full ground truth `State` |
| `/grade` | GET | Final score `{"score": 0.0–1.0}` |
| `/docs` | GET | Interactive Swagger UI |