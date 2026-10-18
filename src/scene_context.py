"""Explicit photographic intent plus conservative count-based scene inference.

Event names are deliberately excluded: a wedding contains portraits, details,
groups and reportage, each requiring different quality criteria.
"""
SHOT_TYPES = ("auto", "portrait", "couple", "group", "candid", "detail", "scene")
MOTION_INTENTS = ("auto", "freeze", "intentional")


def resolve_context(localization, shot_type="auto", motion_intent="auto"):
    if shot_type not in SHOT_TYPES or motion_intent not in MOTION_INTENTS:
        raise ValueError("Unsupported shot type or motion intent")
    people = localization.get("subjects", [])
    count = len(people) if "subjects" in localization else int(localization.get("subject_detected", False))
    inferred = "scene" if count == 0 else "portrait" if count == 1 else "couple" if count == 2 else "group"
    resolved = inferred if shot_type == "auto" else shot_type
    # Automatic detection establishes count, never whether people are posing.
    faces_required = shot_type in ("portrait", "couple", "group")
    reasons = []
    if localization.get("person_detector_status") in ("unavailable", "failed"):
        reasons.append("Person detection unavailable; subject count is unknown.")
    expected = {"portrait": 1, "couple": 2}.get(shot_type)
    if expected and count != expected:
        reasons.append(f"Expected {expected} subjects; detected {count}. Confirm the subjects.")
    if shot_type == "group" and count < 3:
        reasons.append("Expected a group; fewer than three people detected.")
    return {"shot_type": resolved, "source": "inferred" if shot_type == "auto" else "explicit",
            "subject_count": count, "faces_required": faces_required,
            "motion_intent": motion_intent, "review_reasons": reasons}
