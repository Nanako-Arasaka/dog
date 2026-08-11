from obstacle_avoidance.cone_strategy import (
    AvoidanceConfig,
    ConeDetection,
    plan_cone_avoidance,
)


FRAME = (480, 640, 3)


def test_clear_corridor_keeps_moving_forward():
    decision = plan_cone_avoidance([], FRAME)

    assert decision.state == "clear_forward"
    assert decision.vx > 0
    assert decision.wz == 0


def test_close_cone_crawls_instead_of_stopping_in_place():
    decision = plan_cone_avoidance(
        [ConeDetection((220, 80, 420, 420), confidence=0.9)],
        FRAME,
    )

    assert decision.state == "crawl_around"
    assert decision.vx > 0
    assert decision.wz != 0


def test_two_cones_steers_toward_gap_center():
    cfg = AvoidanceConfig(gap_center_deadband_ratio=0.01)
    decision = plan_cone_avoidance(
        [
            ConeDetection((120, 180, 200, 320), confidence=0.9),
            ConeDetection((440, 180, 520, 320), confidence=0.9),
        ],
        FRAME,
        cfg,
    )

    assert decision.state == "gap_follow"
    assert decision.vx > 0
    assert abs(decision.wz) < 1e-6


def test_gap_to_image_right_turns_right():
    cfg = AvoidanceConfig(gap_center_deadband_ratio=0.01)
    decision = plan_cone_avoidance(
        [
            ConeDetection((160, 180, 240, 320), confidence=0.9),
            ConeDetection((520, 180, 600, 320), confidence=0.9),
        ],
        FRAME,
        cfg,
    )

    assert decision.state == "gap_follow"
    assert decision.wz < 0
