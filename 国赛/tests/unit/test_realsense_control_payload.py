from __future__ import annotations

from cone_avoidance.scripts.realsense_aligned_depth_web import make_control_payload


def test_control_payload_includes_latest_pose() -> None:
    stats = {
        "status": "ok",
        "frame": 7,
        "timestamp": 123.0,
        "aligned": True,
        "center_roi": {"min_m": 1.2, "valid_ratio": 0.8},
        "obstacles": [{"x": 0.1, "z": 0.9, "conf": 0.7, "bbox": [1, 2, 3, 4]}],
    }
    pose = {"x": 0.2, "y": -0.1, "yaw": 0.3, "source": "/camera_pose"}

    payload = make_control_payload(stats, fps=15.0, pose=pose)

    assert payload["pose"] == pose
    assert payload["obstacles"][0]["x"] == 0.1
    assert payload["front_depth"] == 1.2
