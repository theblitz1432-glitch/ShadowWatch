"""
ShadowWatch-v0 — FastAPI server
Exposes the OpenEnv interface via HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from env.shadow_env import ShadowWatchEnv
from env.models import Observation, State, StepResult

app = FastAPI(
    title="ShadowWatch-v0",
    description="AI Drone Spy & Defence Environment — OpenEnv Hackathon",
    version="1.0.0",
)

# Singleton environment instance (one per container)
env: ShadowWatchEnv = ShadowWatchEnv("single_target_clear")

VALID_TASKS = ["single_target_clear", "multi_threat_gps_denied", "swarm_electronic_warfare"]


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root():
    return {
        "name":      "ShadowWatch-v0",
        "status":    "running",
        "tasks":     VALID_TASKS,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/docs"],
    }


@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    """
    Reset the environment for the given task. Returns initial Observation.
    Body is optional — defaults to 'single_target_clear' if not provided.
    Accepts: {} | {"task_id": "..."} | no body at all.
    """
    task_id = "single_target_clear"

    try:
        body = await request.json()
        if isinstance(body, dict) and "task_id" in body:
            task_id = body["task_id"]
    except Exception:
        # No body, empty body, or non-JSON — use default task
        pass

    if task_id not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{task_id}'. Must be one of: {VALID_TASKS}"
        )

    obs = env.reset(task_id)
    return obs


@app.post("/step", response_model=StepResult)
def step(body: StepRequest):
    """Execute one action. Returns obs, reward, done, info."""
    try:
        result = env.step(body.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=State)
def get_state():
    """Return full ground truth state (grading / debugging only)."""
    return env.state()


@app.get("/grade")
def grade():
    """Return final episode score in [0.0, 1.0]."""
    score = env.grade()
    return {
        "score":   score,
        "task_id": env.task_id,
        "step":    env.step_count,
        "done":    env.done,
    }