"""Run: python -m streamlit run label_photos.py"""
import os
from pathlib import Path

from PIL import Image, ImageOps
import streamlit as st

from src.annotations import AnnotationStore
from src.scene_context import SHOT_TYPES, MOTION_INTENTS

ROOT = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title="Photographer labels", layout="wide")
    st.title("Photographer labels")
    st.write("Judge suitability for delivery. Deliberate background blur, panning and motion can be Keep. "
             "Use Review for unintended defects or uncertainty, and explain the concern.")
    path = Path(os.environ.get("CONTEXT_IQA_LABELS", ROOT/"outputs"/"human_labels.csv"))
    try:
        store = AnnotationStore(path)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    if not store.rows:
        st.info("The annotation sheet contains no photos.")
        return
    st.session_state.setdefault("shown_digest", store.digest)
    events = sorted({r["event_id"] for r in store.rows})
    reviewer = st.sidebar.text_input("Photographer ID", key="reviewer")
    event = st.sidebar.selectbox("Event", events)
    only_unlabelled = st.sidebar.checkbox("Unlabelled only", value=True)
    labelled = sum(r["expert_score_usability"] in ("Keep", "Review") for r in store.rows)
    st.sidebar.write(f"{labelled} of {len(store.rows)} labelled")
    with st.sidebar.expander("Event partitions"):
        st.caption("Assign whole events. Keep related bursts together. Reserve fresh events for test.")
        split = st.selectbox("Partition", ["train", "validation", "test"])
        if st.button("Assign this whole event"):
            try:
                store.assign_event(event, split, st.session_state["shown_digest"])
                st.session_state.pop("shown_digest", None)
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))
    if st.sidebar.button("Reload annotations"):
        st.session_state.pop("shown_digest", None)
        st.rerun()
    rows = [r for r in store.rows if r["event_id"] == event and
            (not only_unlabelled or not r["expert_score_usability"])]
    if not rows:
        st.success("No unlabelled photos remain in this event.")
        return
    image_id = st.selectbox("Photo", [r["image_id"] for r in rows])
    row = next(r for r in rows if r["image_id"] == image_id)
    if st.session_state.get("shown_photo") != image_id:
        st.session_state["shown_photo"] = image_id
        st.session_state["shown_digest"] = store.digest
    image_path = (ROOT/row["file_path"]).resolve()
    try:
        with Image.open(image_path) as image:
            display = ImageOps.exif_transpose(image).convert("RGB")
        st.image(display, caption=f"{image_id} · {row.get('burst_id', '')}", width="stretch")
    except OSError as exc:
        st.error(f"Cannot open photo: {exc}")
        return
    with st.form("label_"+image_id):
        options = ["Unlabelled", "Keep", "Review"]
        current = row["expert_score_usability"] or "Unlabelled"
        label = st.radio("Photographer judgment", options, index=options.index(current))
        reason = st.text_area("Reason for Review", value=row.get("review_reason", ""))
        notes = st.text_area("Creative intent / notes", value=row.get("creative_notes", ""))
        style_options = ["", "Creative", "Standard"]
        creative_style = st.selectbox("Photographic style", style_options,
            index=style_options.index(row.get("creative_style", "")),
            format_func=lambda value: value or "Unspecified",
            help="Creative effects can be Keep or Review. Judge delivery suitability separately; accidental blur is not a creative effect.")
        st.caption("Use auto context for automatic-use experiments. Explicit context must also be available during assessment.")
        shot = st.selectbox("Shot type", SHOT_TYPES, index=SHOT_TYPES.index(row.get("shot_type") or "auto"))
        motion = st.selectbox("Motion intent", MOTION_INTENTS, index=MOTION_INTENTS.index(row.get("motion_intent") or "auto"))
        if st.form_submit_button("Save judgment"):
            try:
                store.save_label(image_id, label, reviewer, reason, shot, motion, notes,
                                 st.session_state["shown_digest"], creative_style=creative_style)
                st.session_state.pop("shown_digest", None)
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
