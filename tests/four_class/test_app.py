from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("APPLE_ARTIFACTS_DIR", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("No trained classification" in warning.value for warning in app.warning)
    assert not app.selectbox


def test_streamlit_model_discovery():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Apple Disease Explorer"


def test_streamlit_invalid_upload(tmp_path, monkeypatch):
    import io
    import streamlit as st
    upload = io.BytesIO(b"not an image"); upload.size = len(upload.getvalue())
    monkeypatch.setenv("APPLE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py")).run(timeout=30)
    assert not app.exception
    assert any("Cannot decode" in item.value for item in app.error)


def test_streamlit_upload_texture_and_segmentation_without_model(tmp_path, monkeypatch):
    import io
    import numpy as np
    import streamlit as st
    from PIL import Image
    pixels = np.zeros((32, 32, 3), np.uint8)
    pixels[:, :12] = (220, 30, 40); pixels[:, 12:24] = (10, 240, 30); pixels[:, 24:] = (10, 30, 240)
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format="PNG")
    def uploaded(*args, **kwargs):
        value = io.BytesIO(stream.getvalue()); value.size = len(value.getvalue()); return value
    monkeypatch.setenv("APPLE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(st, "file_uploader", uploaded)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app.py")).run(timeout=30)
    assert not app.exception
    assert len(app.dataframe[0].value) == 10
    app.button[0].click().run(timeout=30)
    assert not app.exception
