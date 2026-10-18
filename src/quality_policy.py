"""One scoring policy for inference and parameter optimization."""
def apply_policy(base_score, localization, features, context, config):
    score = base_score
    face_penalty = motion_penalty = 0.0
    reasons = list(context["review_reasons"])
    metrics = features.get("subject_metrics", [])
    if context["faces_required"]:
        if metrics:
            missing = sum(not m["face_detected"] for m in metrics)
            # Detector failure is uncertainty rather than evidence of a bad face.
            if localization.get("face_detector_status") == "ok":
                face_penalty = score * (1-config.face_discount) * missing/len(metrics)
                score -= face_penalty
            if features.get("unassessable_subject_count", 0):
                reasons.append("Some posed subjects have missing or too-small faces; inspect them individually.")
            weak = [m["subject_id"] for m in metrics if m["face_assessable"] and
                    m["local_technical_score"] < config.subject_quality_threshold]
            if weak:
                reasons.append(f"Possible focus problems on posed subjects {weak}; inspect individually.")
        elif not features.get("face_present"):
            reasons.append("Visible faces could not be verified for this posed shot.")
    motion = float(features.get("motion_blur_metric", 0.))
    if motion > config.motion_thresh and context["motion_intent"] != "intentional":
        motion_penalty = min(score, (motion-config.motion_thresh)*config.motion_scale)
        score -= motion_penalty
        if context["motion_intent"] == "auto":
            reasons.append("Possible motion blur; confirm whether movement is intentional.")
    if localization.get("unmatched_face_count", 0):
        reasons.append("Some faces could not be matched to people; subject coverage is incomplete.")
    if context["faces_required"] and any(s.get("possible_head_crop") for s in localization.get("subjects", [])):
        reasons.append("Possible head crop at the image boundary; confirm the intended framing.")
    return max(0., min(100., score)), face_penalty, motion_penalty, reasons
