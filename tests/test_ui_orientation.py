"""The image inspected by a photographer must match assessment coordinates."""
import io

from PIL import Image, ImageOps
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("source_mode", ["Dataset Manifest", "Upload Image"])
def test_sidebar_normalizes_exif_before_preview_and_assessment(tmp_path, monkeypatch, source_mode):
    image = Image.new("RGB", (120, 80), "red")
    image.paste("blue", (60, 0, 120, 80))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path/"rotated.jpg"
    image.save(path, exif=exif)
    from src.ui import components
    monkeypatch.setattr(components.st, "file_uploader", lambda *args, **kwargs: io.BytesIO(path.read_bytes()))
    script = f'''
from pathlib import Path
import streamlit as st
from src.ui.components import render_sidebar_controls
rows = [dict(image_id="rotated", file_path="rotated.jpg", event_id="test")]
image, _, _, _ = render_sidebar_controls(rows, Path({str(tmp_path)!r}))
st.session_state["loaded_image"] = image
'''
    app = AppTest.from_string(script).run()
    if source_mode == "Upload Image":
        app.radio[0].set_value(source_mode).run()
    assert not app.exception
    loaded = app.session_state["loaded_image"]
    with Image.open(path) as original:
        expected = ImageOps.exif_transpose(original).convert("RGB")
    assert loaded.size == (80, 120)
    assert loaded.tobytes() == expected.tobytes()
    assert loaded.getexif().get(274) is None
