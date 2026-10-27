"""Serialización de resultados: PNG, JSON y ZIP (todo en memoria)."""

from __future__ import annotations

import io
import json
import zipfile

import cv2
import numpy as np

from .visualization import to_bgr


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", to_bgr(image))
    if not ok:
        raise ValueError("No se pudo codificar la imagen como PNG.")
    return buf.tobytes()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"No serializable: {type(o)}")


def metrics_json(metrics: dict, homography: np.ndarray) -> bytes:
    payload = {"metrics": metrics, "homography_input_to_template": np.asarray(homography).tolist()}
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")


def homography_json(homography: np.ndarray) -> bytes:
    return json.dumps(
        {"homography_input_to_template": np.asarray(homography).tolist()}, indent=2
    ).encode("utf-8")


def build_files(*, aligned, matches_img, overlay, differences, metrics: dict, homography) -> dict[str, bytes]:
    """Devuelve ``{nombre: bytes}`` solo con archivos que realmente existen."""
    files = {
        "aligned.png": png_bytes(aligned),
        "matches.png": png_bytes(matches_img),
        "overlay.png": png_bytes(overlay),
        "metrics.json": metrics_json(metrics, homography),
        "homography.json": homography_json(homography),
    }
    if differences is not None:
        files["differences.png"] = png_bytes(differences)
    return files


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)  # nombres fijos: nunca derivados del usuario
    return buf.getvalue()
