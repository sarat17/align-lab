import cv2
import numpy as np
import pytest

from src import detectors, matching


class FakeMatch:
    def __init__(self, distance):
        self.distance = distance


def test_ratio_test_tolerates_short_lists():
    knn = [[FakeMatch(1.0)], [], [FakeMatch(1.0), FakeMatch(10.0)], [FakeMatch(9.0), FakeMatch(10.0)]]
    good = matching.ratio_test(knn, 0.75)
    assert len(good) == 1 and good[0].distance == 1.0


def test_matcher_follows_detector_name():
    assert isinstance(matching.create_matcher("ORB"), cv2.BFMatcher)
    assert isinstance(matching.create_matcher("SIFT"), cv2.BFMatcher)
    with pytest.raises(matching.MatchingError):
        matching.create_matcher("SURF")


def test_none_descriptors_give_clear_error():
    with pytest.raises(matching.MatchingError, match="descriptores"):
        matching.match_descriptors(None, np.zeros((5, 32), np.uint8), "ORB")


def test_too_few_descriptors_give_clear_error():
    with pytest.raises(matching.MatchingError):
        matching.match_descriptors(np.zeros((1, 32), np.uint8), np.zeros((5, 32), np.uint8), "ORB")


def test_invalid_ratio_rejected():
    d = np.zeros((5, 32), np.uint8)
    with pytest.raises(matching.MatchingError):
        matching.match_descriptors(d, d, "ORB", ratio=1.5)


def test_orb_self_matching_finds_many_matches(template):
    res = detectors.detect_and_describe(template, "ORB", 1000)
    good = matching.match_descriptors(res.descriptors, res.descriptors, "ORB", 0.9)
    assert len(good) > 50


def test_select_matches_orders_and_filters():
    ms = [FakeMatch(d) for d in (5, 1, 3, 2, 4, 9, 8, 7, 6, 10)]
    assert [m.distance for m in matching.select_matches(ms, 1.0)] == sorted(m.distance for m in ms)
    kept = matching.select_matches(ms, 0.5)
    assert [m.distance for m in kept] == [1, 2, 3, 4, 5]
    with pytest.raises(matching.MatchingError):
        matching.select_matches(ms, 0.0)
