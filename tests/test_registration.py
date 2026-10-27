import cv2
import numpy as np
import pytest

from src import detectors, export, metrics, registration
from src.pipeline import AlignParams, ImageLoadError, align_images, compare_detectors, decode_image, validate_params
from conftest import H_TRUE


class DM:  # cv2.DMatch mínimo
    def __init__(self, q, t):
        self.queryIdx, self.trainIdx, self.distance = q, t, 0.0


class KP:
    def __init__(self, pt):
        self.pt = pt


def _synthetic_matches(n=60, outliers=10, seed=1):
    rng = np.random.default_rng(seed)
    src = rng.uniform(0, 600, (n, 2)).astype(np.float32)
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_TRUE).reshape(-1, 2).astype(np.float32)
    dst[:outliers] += rng.uniform(80, 200, (outliers, 2)).astype(np.float32)  # outliers gruesos
    return [KP(tuple(p)) for p in src], [KP(tuple(p)) for p in dst], [DM(i, i) for i in range(n)]


def test_homography_recovered_from_known_points():
    kin, ktp, ms = _synthetic_matches()
    reg = registration.estimate_homography(ms, kin, ktp, 3.0, image_shape=(480, 640), template_shape=(480, 640))
    Hn = reg.H / reg.H[2, 2]
    assert np.allclose(Hn, H_TRUE, atol=1e-2, rtol=1e-2)
    assert reg.n_inliers >= 45
    assert not reg.inlier_mask[:10].any()  # los outliers inyectados se descartan


def test_fewer_than_four_matches_is_an_error():
    kin, ktp, ms = _synthetic_matches()
    with pytest.raises(registration.RegistrationError, match="al menos 4"):
        registration.estimate_homography(ms[:3], kin, ktp)


def test_degenerate_and_invalid_homographies_rejected():
    shape = (480, 640)
    with pytest.raises(registration.RegistrationError):
        registration.validate_homography(None, shape, shape)
    bad = np.eye(3)
    bad[0, 0] = np.nan
    with pytest.raises(registration.RegistrationError):
        registration.validate_homography(bad, shape, shape)
    collapse = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1.0]])
    with pytest.raises(registration.RegistrationError, match="degenerada"):
        registration.validate_homography(collapse, shape, shape)
    mirror = np.array([[-1, 0, 640], [0, 1, 0], [0, 0, 1.0]])
    with pytest.raises(registration.RegistrationError, match="reflexión|orientación"):
        registration.validate_homography(mirror, shape, shape)


def test_warp_has_template_size_and_coverage_mask(distorted, template):
    out, cov = registration.warp_to_template(distorted, H_TRUE, template.shape)
    assert out.shape == template.shape
    assert cov.shape == template.shape[:2] and cov.dtype == np.uint8
    assert 0 < (cov > 0).mean() <= 1.0


@pytest.mark.parametrize("det", ["SIFT", "ORB"])
def test_full_pipeline_recovers_synthetic_transform(distorted, template, det):
    if det == "SIFT" and not detectors.sift_available():
        pytest.skip("SIFT no disponible")
    r = align_images(distorted, template, AlignParams(detector=det))
    assert r.aligned.shape == template.shape
    corners = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]]).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, r.H).reshape(-1, 2)
    true = cv2.perspectiveTransform(corners, H_TRUE).reshape(-1, 2)
    assert np.linalg.norm(est - true, axis=1).mean() < 3.0
    m = r.metrics
    assert m["inliers"] + m["outliers"] == m["matches_used_for_homography"]
    assert m["detector_used"] == det
    assert 0 < m["inlier_rate"] <= 1
    assert m["reprojection_error_inliers_px"]["mean_px"] < 3.0


def test_pipeline_error_on_featureless_input(template):
    blank = np.full((300, 300, 3), 128, np.uint8)
    with pytest.raises(Exception, match="pocos puntos clave"):
        align_images(blank, template, AlignParams(detector="ORB"))


def test_pipeline_accepts_gray_and_bgra(distorted, template):
    g = align_images(cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY), cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), AlignParams(detector="ORB"))
    a = align_images(cv2.cvtColor(distorted, cv2.COLOR_BGR2BGRA), template, AlignParams(detector="ORB"))
    assert g.aligned.ndim == 3 and a.aligned.ndim == 3


def test_param_validation():
    with pytest.raises(ValueError):
        validate_params(AlignParams(ratio=2.0))
    with pytest.raises(ValueError):
        validate_params(AlignParams(ransac_threshold=0.0))
    with pytest.raises(ValueError):
        validate_params(AlignParams(max_features=5))


def test_decode_image_errors_and_limits(template):
    ok, buf = cv2.imencode(".png", template)
    img, info = decode_image(buf.tobytes(), "a.png")
    assert info["width"] == 640 and info["channels"] == 3
    with pytest.raises(ImageLoadError):
        decode_image(b"", "a.png")
    with pytest.raises(ImageLoadError, match="dañado|válida"):
        decode_image(b"no soy una imagen", "a.png")
    with pytest.raises(ImageLoadError, match="no soportado"):
        decode_image(buf.tobytes(), "a.gif")
    with pytest.raises(ImageLoadError, match="límite"):
        decode_image(buf.tobytes(), "a.png", max_pixels=1000)
    with pytest.raises(ImageLoadError, match="límite"):
        decode_image(buf.tobytes(), "a.png", max_file_mb=0.001)


def test_difference_stats_only_use_valid_pixels():
    a = np.zeros((10, 10), np.uint8)
    b = np.full((10, 10), 100, np.uint8)
    cov = np.zeros((10, 10), np.uint8)
    cov[:5] = 255
    s = metrics.difference_stats(a, b, cov)
    assert s["valid_pixels"] == 50 and s["mean_abs_diff"] == 100.0
    assert metrics.difference_stats(a, b, np.zeros_like(cov))["mean_abs_diff"] is None


def test_export_matches_run_and_contains_real_values(distorted, template):
    r = align_images(distorted, template, AlignParams(detector="ORB"))
    files = export.build_files(aligned=r.aligned, matches_img=r.matches_img, overlay=r.overlay,
                               differences=r.differences, metrics=r.metrics, homography=r.H)
    assert {"aligned.png", "matches.png", "overlay.png", "differences.png", "metrics.json", "homography.json"} <= set(files)
    import json, io, zipfile
    payload = json.loads(files["metrics.json"])
    assert payload["metrics"]["inliers"] == r.metrics["inliers"]
    assert np.allclose(payload["homography_input_to_template"], r.H)
    assert set(zipfile.ZipFile(io.BytesIO(export.zip_bytes(files))).namelist()) == set(files)
    no_diff = export.build_files(aligned=r.aligned, matches_img=r.matches_img, overlay=r.overlay,
                                 differences=None, metrics=r.metrics, homography=r.H)
    assert "differences.png" not in no_diff


def test_compare_detectors_reports_failures_not_zeros(template):
    blank = np.full((300, 300, 3), 128, np.uint8)
    rows = compare_detectors(blank, template, AlignParams())
    assert all(r["status"] == "falló" and r["error"] for r in rows)
    assert all("inliers" not in r for r in rows)


def test_robust_alignment_with_template_mask():
    import cv2
    import numpy as np
    from src.pipeline import AlignParams, align_images_robust

    rng = np.random.default_rng(5)
    tpl = np.full((500, 700, 3), 255, np.uint8)
    for _ in range(300):
        c = tuple(int(v) for v in rng.integers(0, [700, 500]))
        cv2.circle(tpl, c, int(rng.integers(3, 10)), tuple(int(v) for v in rng.integers(0, 200, 3)), -1)
    M = cv2.getRotationMatrix2D((350, 250), 12, 0.9)
    img = cv2.warpAffine(tpl, M, (700, 500), borderValue=(255, 255, 255))
    mask = np.full(tpl.shape[:2], 255, np.uint8)
    mask[200:300, 200:500] = 0
    res = align_images_robust(img, tpl, AlignParams(), template_mask=mask)
    assert res.metrics["inliers"] >= 15
    assert res.aligned.shape == tpl.shape
