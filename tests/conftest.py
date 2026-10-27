"""Fixtures sintéticas compartidas: no dependen de archivos personales."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H_TRUE = np.array([[0.95, 0.12, 30.0], [-0.08, 1.02, 20.0], [1e-4, 5e-5, 1.0]], dtype=np.float64)


def make_template(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 235, np.uint8)
    for _ in range(120):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if rng.random() < 0.5:
            cv2.circle(img, p, int(rng.integers(5, 30)), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + int(rng.integers(10, 60)), p[1] + int(rng.integers(10, 60))), c, -1)
    cv2.putText(img, "ALIGN LAB", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 20, 20), 4)
    return img


@pytest.fixture(scope="session")
def template():
    return make_template()


@pytest.fixture(scope="session")
def distorted(template):
    """Plantilla deformada con la inversa de H_TRUE: alinear debe recuperar H_TRUE."""
    h, w = template.shape[:2]
    return cv2.warpPerspective(template, np.linalg.inv(H_TRUE), (w, h))
