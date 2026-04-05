"""
ShadowWatch-v0 — Task graders.
All graders return float in [0.0, 1.0].
Scores vary with agent performance — never constant.
"""

from env.models import State


def grade_task1(state: State) -> float:
    """
    Task 1 — Single Target, Clear Signal
    score = detection_accuracy × speed_bonus × battery_remaining
    """
    threats = [t for t in state.all_threats if not t.is_decoy]
    if not threats:
        return 0.0

    confirmed          = sum(1 for t in threats if t.confirmed)
    detection_accuracy = confirmed / len(threats)           # 0.0 or 1.0 (1 threat)

    speed_bonus = max(0.0, 1.0 - state.step / 40.0) + 0.5  # faster = higher bonus
    speed_bonus = min(1.0, speed_bonus)

    active           = next((d for d in state.all_drones if d.is_active), state.all_drones[0])
    battery_remaining = active.battery

    score = detection_accuracy * speed_bonus * battery_remaining
    return round(min(1.0, max(0.0, score)), 4)


def grade_task2(state: State) -> float:
    """
    Task 2 — Multi-Threat, GPS Denied
    score = threats_found_ratio × location_accuracy × (1 − false_alarm_rate)
    """
    real_threats = [t for t in state.all_threats if not t.is_decoy]
    if not real_threats:
        return 0.0

    found               = sum(1 for t in real_threats if t.confirmed)
    threats_found_ratio = found / len(real_threats)

    location_accuracy   = 0.85 if found > 0 else 0.0   # proxy: scan radius guarantee

    sb             = state.score_breakdown
    correct_alerts = max(0, int(round(sb.alert_reward)))       # +1.0 per correct alert
    # penalty_total accumulates 0.01/step + 0.40/false_alarm + other costs
    # isolate false alarms: subtract step cost and other penalties
    step_cost      = state.step * 0.01
    other_penalty  = sb.penalty_total - step_cost
    false_alarms   = max(0, int(round(other_penalty / 0.40)))
    total_attempts = correct_alerts + false_alarms
    false_alarm_rate = (false_alarms / total_attempts) if total_attempts > 0 else 0.0

    score = threats_found_ratio * location_accuracy * (1.0 - false_alarm_rate)
    return round(min(1.0, max(0.0, score)), 4)


def grade_task3(state: State) -> float:
    """
    Task 3 — Swarm + Decoys + Electronic Warfare
    score = detection_recall × decoy_rejection × report_quality × coordination
    """
    real_threats = [t for t in state.all_threats if not t.is_decoy]
    decoys       = [t for t in state.all_threats if t.is_decoy]

    # Detection recall
    confirmed_real   = sum(1 for t in real_threats if t.confirmed)
    detection_recall = confirmed_real / len(real_threats) if real_threats else 0.0

    # Decoy rejection (decoys should NOT be confirmed)
    wrong_decoys    = sum(1 for t in decoys if t.confirmed)
    decoy_rejection = 1.0 - (wrong_decoys / len(decoys)) if decoys else 1.0

    # Report quality (correct alerts vs possible)
    max_alert_reward = len(real_threats) * 1.0
    report_quality   = (
        min(1.0, state.score_breakdown.alert_reward / max_alert_reward)
        if max_alert_reward > 0 else 0.0
    )

    # Coordination efficiency (floor at 0.1 so it doesn't zero the score)
    max_coord    = max(state.step * 0.075, 1.0)   # 0.15 * 0.5 upper bound
    coordination = min(1.0, state.score_breakdown.coordination_bonus / max_coord)
    coordination = max(0.1, coordination)

    score = detection_recall * decoy_rejection * report_quality * coordination
    return round(min(1.0, max(0.0, score)), 4)