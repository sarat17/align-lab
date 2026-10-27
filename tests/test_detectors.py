import cv2
import numpy as np
import pytest

from src import detectors


def test_orb_detects_keypoints(template):
    res = detectors.detect_and_describe(template, "ORB", 2000)
    assert res.detector == "ORB"
    assert res.count >= 4
    assert res.descriptors.dtype == np.uint8  # binarios -> Hamming


def test_sift_when_available(template):
    if not detectors.sift_available():
        pytest.skip("SIFT no disponible en esta instalación")
    res = detectors.detect_and_describe(template, "SIFT", 2000)
    assert res.detector == "SIFT"
    assert res.descriptors.dtype == np.float32  # flotantes -> L2


def test_resolve_falls_back_to_orb_and_reports_it(monkeypatch):
    monkeypatch.setattr(detectors, "sift_available", lambda: False)
    assert detectors.resolve_detector_name("SIFT") == "ORB"
    assert detectors.available_detectors() == ["ORB"]


def test_unknown_detector_rejected():
    with pytest.raises(detectors.DetectionError):
        detectors.resolve_detector_name("SURF")


def test_blank_image_gives_clear_error():
    blank = np.full((200, 200, 3), 128, np.uint8)
    with pytest.raises(detectors.DetectionError, match="pocos puntos clave"):
        detectors.detect_and_describe(blank, "ORB", 500)


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_to_gray_handles_all_channel_layouts(template, channels):
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if channels is None:
        img = gray
    elif channels == 1:
        img = gray[:, :, None]
    elif channels == 3:
        img = template
    else:
        img = cv2.cvtColor(template, cv2.COLOR_BGR2BGRA)
    out = detectors.to_gray(img)
    assert out.ndim == 2 and out.dtype == np.uint8
