import unittest

import cv2
import numpy as np

from motion_detector import DetectorConfig, MotionDetector, SalientTargetSelector


class MotionDetectorTests(unittest.TestCase):
    def test_detects_synthetic_moving_rectangle_after_warmup(self) -> None:
        detector = MotionDetector(
            DetectorConfig(
                history=30,
                var_threshold=16,
                warmup_frames=12,
                min_area=100,
                open_size=3,
                close_size=5,
                dilate_iterations=1,
            )
        )

        background = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(15):
            detector.detect(background)

        frame = background.copy()
        cv2.rectangle(frame, (80, 90), (130, 160), (255, 255, 255), -1)
        _, detections = detector.detect(frame)

        self.assertTrue(detections)
        largest = detections[0]
        self.assertGreaterEqual(largest.area, 100)
        self.assertLess(largest.x, 100)
        self.assertGreater(largest.x + largest.width, 120)

    def test_selector_keeps_nearby_target(self) -> None:
        from motion_detector import Detection

        selector = SalientTargetSelector(max_jump_ratio=0.3)
        first = Detection(10, 20, 80, 80, 6400)
        self.assertIs(selector.select([first], (300, 400, 3)), first)

        nearby = Detection(20, 25, 70, 70, 4900)
        far_larger = Detection(280, 180, 100, 100, 10000)
        self.assertIs(selector.select([nearby, far_larger], (300, 400, 3)), nearby)

    def test_box_area_is_distinct_from_motion_pixels(self) -> None:
        from motion_detector import Detection

        detection = Detection(
            10, 20, 100, 200, 1500, label="person", motion_ratio=0.075
        )
        self.assertEqual(detection.area, 1500)
        self.assertEqual(detection.box_area, 20000)


if __name__ == "__main__":
    unittest.main()
