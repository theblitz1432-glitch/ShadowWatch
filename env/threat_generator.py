"""
ShadowWatch-v0 — Threat and fog zone generation.
"""

import random
from typing import Optional


TASK_CONFIG = {
    "single_target_clear": {
        "threats": [
            {"threat_type": "infiltration", "is_decoy": False},
        ],
        "fog_zones": False,
        "mobile": False,
    },
    "multi_threat_gps_denied": {
        "threats": [
            {"threat_type": "military_camp", "is_decoy": False},
            {"threat_type": "convoy",        "is_decoy": False},
            {"threat_type": "bunker",        "is_decoy": False},
        ],
        "fog_zones": True,
        "mobile": False,
    },
    "swarm_electronic_warfare": {
        "threats": [
            {"threat_type": "military_camp", "is_decoy": False},
            {"threat_type": "convoy",        "is_decoy": False},
            {"threat_type": "bunker",        "is_decoy": False},
            {"threat_type": "infiltration",  "is_decoy": True},
            {"threat_type": "convoy",        "is_decoy": True},
        ],
        "fog_zones": True,
        "mobile": True,
    },
}


def _random_position(grid_size: int, occupied: list) -> list:
    while True:
        pos = [random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)]
        if pos not in occupied:
            return pos


def _generate_fog_zones(grid_size: int, n_zones: int = 3) -> list:
    return [
        {
            "center":  [random.randint(2, grid_size - 3), random.randint(2, grid_size - 3)],
            "radius":  random.randint(2, 4),
            "density": random.choice(["partial_fog", "heavy_fog"]),
        }
        for _ in range(n_zones)
    ]


def generate_threats(task_id: str, grid_size: Optional[int] = None) -> dict:
    """Generate threats, decoys, and fog zones for the given task."""
    if task_id not in TASK_CONFIG:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIG.keys())}")

    config   = TASK_CONFIG[task_id]
    gs       = grid_size if grid_size is not None else 10
    occupied = []
    threats  = []

    for t in config["threats"]:
        pos = _random_position(gs, occupied)
        occupied.append(pos)
        threats.append({
            "threat_type": t["threat_type"],
            "position":    pos,
            "is_decoy":    t["is_decoy"],
            "detected":    False,
            "confirmed":   False,
            "alerted":     False,
        })

    fog_zones = (
        _generate_fog_zones(gs, 2 if task_id == "multi_threat_gps_denied" else 4)
        if config["fog_zones"] else []
    )

    return {
        "threats":   threats,
        "fog_zones": fog_zones,
        "mobile":    config["mobile"],
        "grid_size": gs,
    }


def move_threats(threats: list, grid_size: int) -> list:
    """Move non-decoy threats one step randomly. Called every 10 steps in Task 3."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for t in threats:
        if not t["is_decoy"]:
            dr, dc = random.choice(directions)
            r, c   = t["position"]
            t["position"] = [
                max(0, min(grid_size - 1, r + dr)),
                max(0, min(grid_size - 1, c + dc)),
            ]
    return threats