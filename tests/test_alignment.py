import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = unittest.TestCase()


def test_causal_window_boundary_is_half_open():
    from label_transfer import event_index_bounds
    timestamps = np.array([99, 100, 150, 199, 200])
    lo, hi = event_index_bounds(timestamps, 100, 200)
    assert timestamps[lo:hi].tolist() == [100, 150, 199]


def _tracks(box_at_k, box_at_future):
    return {"track": {"class_id": 0, "by_frame": {
        0: {"cx": box_at_k, "cy": 0.5, "w": 0.2, "h": 0.1},
        1: {"cx": box_at_future, "cy": 0.5, "w": 0.2, "h": 0.1},
    }}}


def test_frame_k_label_has_no_future_frame_influence():
    from label_transfer import build_causal_windows
    times = np.array([1000, 2000])
    first = build_causal_windows(times, _tracks(0.25, 0.50), 1000)
    changed_future = build_causal_windows(times, _tracks(0.25, 0.95), 1000)
    assert first[0]["boxes"][0] == changed_future[0]["boxes"][0]
    assert first[0]["boxes"][0]["cx"] == 0.25


def test_fixed_count_scale_preserves_amplitude_below_clip():
    from build_event_dataset import encode_count_u8
    one = int(encode_count_u8(np.array([2]), 10)[0])
    doubled = int(encode_count_u8(np.array([4]), 10)[0])
    assert abs(doubled - 2 * one) <= 1


def test_native_shape_and_letterbox_preserve_rectangle_ratio():
    from build_event_dataset import letterbox_image, transform_box_letterbox
    native = np.zeros((720, 1280, 3), np.uint8)
    assert native.shape == (720, 1280, 3)
    boxed, transform = letterbox_image(native, 640, 640)
    result = transform_box_letterbox(
        {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.2}, transform)
    source_ratio = (0.4 * 1280) / (0.2 * 720)
    output_ratio = (result["w"] * 640) / (result["h"] * 640)
    assert boxed.shape == (640, 640, 3)
    assert np.isclose(output_ratio, source_ratio)


def test_linear_primary_conversion_reference_vectors():
    from linear16_to_srgb8 import convert_linear_primaries
    identity = np.array([[0.1, 0.2, 0.3]], np.float64)
    assert np.array_equal(convert_linear_primaries(identity, "srgb"), identity)
    neutral = convert_linear_primaries(np.array([[0.5, 0.5, 0.5]]), "bt2020")
    assert np.isclose(neutral[0, 0], neutral[0, 1], atol=2e-10)
    assert np.isclose(neutral[0, 1], neutral[0, 2], atol=2e-10)
    expected = np.array([1.66049100, -0.12455047, -0.01815076])
    actual = convert_linear_primaries(np.array([[1.0, 0.0, 0.0]]), "bt2020")[0]
    assert np.allclose(actual, expected, atol=2e-7)


def test_bias_text_round_trip_and_range_guard():
    from generate_biases import DEFAULT_BIASES, format_bias_text, parse_bias_text
    values = {**DEFAULT_BIASES, "bias_diff_on": 20, "bias_fo": -5, "bias_hpf": 12}
    assert parse_bias_text(format_bias_text(values)) == values
    with CASE.assertRaises(ValueError):
        format_bias_text({**values, "bias_hpf": -1})


def test_cached_event_intrinsics_without_rms(tmp_path):
    from calibrate_event_camera import load_cached_event_intrinsics
    path = tmp_path / "event_camera_params.npz"
    np.savez(path, K_event=np.eye(3), D_event=np.zeros(5))
    K, D, rms = load_cached_event_intrinsics(path)
    assert np.array_equal(K, np.eye(3))
    assert np.array_equal(D, np.zeros(5))
    assert rms is None
    assert json.loads(json.dumps({"rms_event_intrinsic_px": rms}))[
        "rms_event_intrinsic_px"] is None


def test_rgb_event_offset_sign_convention():
    from estimate_rgb_event_offset import estimate_offset_from_signals
    times = np.arange(0, 200_000, 1_000, dtype=float)
    rgb = ((times // 20_000) % 2).astype(float)
    injected = 7_000.0
    result = estimate_offset_from_signals(times, rgb, times + injected, rgb)
    assert abs(result["offset_us"] - injected) <= 1_000


def _import_sam_export_without_sam3():
    fake_builder = types.ModuleType("sam3.model_builder")
    fake_builder.build_sam3_video_predictor = lambda: None
    sys.modules.setdefault("sam3", types.ModuleType("sam3"))
    sys.modules["sam3.model_builder"] = fake_builder
    sys.modules.pop("sam3_export_tracks", None)
    return importlib.import_module("sam3_export_tracks")


def test_bidirectional_merge_is_deterministic_and_receding_is_only_tiebreak():
    module = _import_sam_export_without_sam3()
    forward = {"f": {"class_id": 0, "class_name": "car", "frames": [
        {"frame_idx": 0, "cx": .5, "cy": .7, "w": .4, "h": .3,
         "score": .8, "source": "forward"},
        {"frame_idx": 1, "cx": .5, "cy": .5, "w": .2, "h": .15,
         "score": .8, "source": "forward"},
    ]}}
    backward = {"b": {"class_id": 0, "class_name": "car", "frames": [
        {"frame_idx": 0, "cx": .5, "cy": .7, "w": .4, "h": .3,
         "score": .8, "source": "backward"},
        {"frame_idx": 1, "cx": .5, "cy": .51, "w": .21, "h": .16,
         "score": .8, "source": "backward"},
    ]}}
    config = {"horizon_y": .45, "continuity_iou_tolerance": .02, "score_tolerance": .02}
    first = module.merge_directional_tracks(forward, backward, config)
    second = module.merge_directional_tracks(forward, backward, config)
    assert first == second
    reasons = [row["merge_reason"] for entry in first.values() for row in entry["frames"]]
    assert reasons
    assert set(reasons) <= {"continuity", "confidence", "receding_tiebreak",
                            "deterministic_forward_tie", "only_forward", "only_backward"}


def test_baseline_metric_api_matches_original_score():
    from calibrate_simulator import score_candidate, score_metric_vectors
    real = {"rate_hz": 100.0, "n_on": 60, "n": 100}
    sim = {"rate_hz": 125.0, "n_on": 65, "n": 100}
    original = score_candidate(sim, real)
    vector_score = score_metric_vectors(
        {"rate_total": 125.0, "on_fraction": 0.65},
        {"rate_total": 100.0, "on_fraction": 0.60},
        {"rate_total": 1.0, "on_fraction": 0.5})
    assert np.isclose(vector_score, original)


def test_condition_roots_are_disjoint_and_manifest_guarded():
    script = (ROOT / "run_pipeline.sh").read_text()
    assert 'DATASET_BASE="data/datasets"' in script
    assert 'ROOT="${DATASET_BASE}/${SIMULATOR_NAME}/${CONDITION}/${TAG}"' in script
    assert 'data/dataset"' not in script
    assert "dataset_manifest.json" in script


if __name__ == "__main__":
    failures = []
    for name, function in sorted(globals().copy().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            if name == "test_cached_event_intrinsics_without_rms":
                with tempfile.TemporaryDirectory() as directory:
                    function(Path(directory))
            else:
                function()
            print(f"PASS {name}")
        except Exception as error:
            failures.append((name, error))
            print(f"FAIL {name}: {error}")
    if failures:
        raise SystemExit(1)
