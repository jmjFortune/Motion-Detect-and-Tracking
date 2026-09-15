"""Synthetic, real-OpenCV regression tests for shared background motion."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

APP_ROOT = str(Path(__file__).resolve().parent.parent)
if APP_ROOT not in sys.path:  # Tests live in test/ but import the root modules.
    sys.path.insert(0, APP_ROOT)

from camera_motion import CameraMotionResult, SharedCameraMotion


EMPTY = np.empty((0, 4), dtype=np.float32)
IDENTITY = np.float32([[1, 0, 0], [0, 1, 0]])


def texture(height=240, width=320, seed=42):
    return np.random.default_rng(seed).integers(
        0, 180, (height, width, 3), dtype=np.uint8
    )


def translated(frame, dx=7, dy=4):
    return cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (frame.shape[1], frame.shape[0]))


class SharedCameraMotionTests(unittest.TestCase):
    # Each expectation catches a consumer-visible break: false reliability,
    # incorrect direction/units, foreground contamination, or stale history.
    def pair(self, previous, current, previous_boxes=EMPTY, current_boxes=EMPTY,
             **kwargs):
        camera = SharedCameraMotion(**kwargs)
        camera.apply(previous, previous_boxes)
        warp = camera.apply(current, current_boxes)
        self.assertIs(warp, camera.last_result.warp)
        return camera, camera.last_result

    def assert_quiet(self, result, limit=0.01):
        self.assertGreater(np.count_nonzero(result.valid_mask), 0)
        ratio = np.count_nonzero(result.motion_mask) / np.count_nonzero(result.valid_mask)
        self.assertLess(ratio, limit)
        self.assertFalse(result.motion_mask[result.valid_mask == 0].any())

    def test_translation_reliable_full_frame_masks_and_invalid_new_view(self):
        frame = texture()
        camera, result = self.pair(frame, translated(frame))
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, [[1, 0, 7], [0, 1, 4]], atol=0.7)
        self.assertEqual(result.warp.dtype, np.float32)
        for mask in (result.motion_mask, result.valid_mask):
            self.assertEqual(mask.shape, (240, 320))
            self.assertEqual(mask.dtype, np.uint8)
        self.assertFalse(result.valid_mask[:, :7].any())
        self.assertFalse(result.valid_mask[:4, :].any())
        self.assertGreaterEqual(result.tracked_points, 12)
        self.assertGreater(result.inlier_ratio, 0.6)
        self.assertGreater(result.coverage, 0.1)
        self.assert_quiet(result)
        self.assertEqual(camera.method, 'sparseOptFlow')
        self.assertEqual(camera.calls, 2)

    def test_first_frame_is_unknown_not_stationary(self):
        camera = SharedCameraMotion()
        warp = camera.apply(texture())
        np.testing.assert_array_equal(warp, IDENTITY)
        self.assertFalse(camera.last_result.reliable)
        self.assertTrue(camera.last_result.reason)
        self.assertFalse(camera.last_result.valid_mask.any())
        self.assertFalse(camera.last_result.motion_mask.any())

    def test_stationary_texture_is_trustworthy_identity(self):
        frame = texture()
        _, result = self.pair(frame, frame.copy())
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, IDENTITY, atol=0.05)
        self.assert_quiet(result)

    def test_textureless_scene_returns_unreliable_identity(self):
        frame = np.full((240, 320, 3), 90, np.uint8)
        _, result = self.pair(frame, frame.copy())
        self.assertFalse(result.reliable)
        self.assertTrue(result.reason)
        np.testing.assert_array_equal(result.warp, IDENTITY)
        self.assertFalse(result.motion_mask.any())
        self.assertFalse(result.valid_mask.any())

    def test_previous_xyxy_foreground_does_not_dominate(self):
        # A moving textured object covers most pixels; dim background is real
        # texture but would lose unrestricted corner ranking to the object.
        background = (texture() // 6).astype(np.uint8)
        previous = background.copy()
        previous[25:215, 35:285] = texture(190, 250, seed=10)
        current = translated(background)
        current[25:215, 47:297] = previous[25:215, 35:285]
        _, result = self.pair(previous, current, [[35, 25, 285, 215]],
                              [[47, 25, 297, 215]])
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, [[1, 0, 7], [0, 1, 4]], atol=0.7)

    def test_current_foreground_excludes_background_tracks(self):
        # Previously unlabelled strong features become a labelled object.
        previous = (texture() // 6).astype(np.uint8)
        previous[25:215, 35:285] = texture(190, 250, seed=10)
        current = translated(previous)
        current[25:215, 47:297] = previous[25:215, 35:285]
        _, result = self.pair(previous, current, EMPTY, [[25, 15, 310, 230]])
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, [[1, 0, 7], [0, 1, 4]], atol=0.7)

    def test_brightness_offset_is_removed_from_residual(self):
        frame = texture()
        current = np.clip(translated(frame).astype(np.int16) + 35, 0, 255).astype(np.uint8)
        _, result = self.pair(frame, current)
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, [[1, 0, 7], [0, 1, 4]], atol=0.7)
        self.assert_quiet(result)

    def test_resized_warp_is_restored_to_input_coordinates(self):
        frame = texture(481, 853)
        _, result = self.pair(frame, translated(frame, 14, 8), max_width=320)
        self.assertTrue(result.reliable, result.reason)
        np.testing.assert_allclose(result.warp, [[1, 0, 14], [0, 1, 8]], atol=0.7)
        self.assertEqual(result.valid_mask.shape, (481, 853))
        self.assertFalse(result.valid_mask[:, :14].any())
        self.assertFalse(result.valid_mask[:8].any())
        self.assert_quiet(result, limit=0.03)

    def test_local_change_survives_semantic_exclusion_and_brightness_removal(self):
        frame = texture()
        current = frame.copy()
        current[85:145, 115:175] = 240
        _, result = self.pair(frame, current, [[100, 70, 190, 160]],
                              [[100, 70, 190, 160]])
        self.assertTrue(result.reliable, result.reason)
        self.assertGreater(result.motion_ratio((115, 85, 175, 145)), 0.9)
        self.assertLess(result.motion_ratio((10, 10, 80, 60)), 0.01)

    def test_clustered_background_points_are_not_reliable(self):
        frame = np.full((240, 320, 3), 90, np.uint8)
        frame[85:135, 135:185] = texture(50, 50)
        _, result = self.pair(frame, translated(frame))
        self.assertFalse(result.reliable)
        self.assertGreaterEqual(result.tracked_points, 12)
        self.assertFalse(result.motion_mask.any())

    def test_too_few_points_and_fully_excluded_background_are_unknown(self):
        frame = texture()
        for kwargs, boxes in (({'min_points': 10000}, EMPTY),
                              ({}, [[-20, -20, 400, 300]])):
            with self.subTest(kwargs=kwargs, boxes=boxes):
                _, result = self.pair(frame, translated(frame), boxes, boxes, **kwargs)
                self.assertFalse(result.reliable)
                np.testing.assert_array_equal(result.warp, IDENTITY)

    def test_unrelated_frames_reject_bad_correspondence(self):
        _, result = self.pair(texture(), texture(seed=90))
        self.assertFalse(result.reliable)
        np.testing.assert_array_equal(result.warp, IDENTITY)

    def test_implausible_scaling_is_rejected(self):
        frame = texture()
        warp = cv2.getRotationMatrix2D((160, 120), 0, 1.5)
        current = cv2.warpAffine(frame, warp, (320, 240))
        _, result = self.pair(frame, current)
        self.assertFalse(result.reliable)
        np.testing.assert_array_equal(result.warp, IDENTITY)

    def test_failure_stores_current_frame_for_next_pair(self):
        camera = SharedCameraMotion()
        camera.apply(np.full((240, 320, 3), 90, np.uint8))
        frame = texture()
        camera.apply(frame)
        self.assertFalse(camera.last_result.reliable)
        camera.apply(translated(frame))
        self.assertTrue(camera.last_result.reliable, camera.last_result.reason)
        np.testing.assert_allclose(camera.last_result.warp, [[1, 0, 7], [0, 1, 4]], atol=0.7)

    def test_shape_change_reinitializes_then_recovers(self):
        camera = SharedCameraMotion()
        camera.apply(texture())
        frame = texture(180, 260)
        camera.apply(frame)
        self.assertFalse(camera.last_result.reliable)
        self.assertEqual(camera.last_result.valid_mask.shape, (180, 260))
        camera.apply(translated(frame))
        self.assertTrue(camera.last_result.reliable, camera.last_result.reason)

    def test_reset_clears_history_without_changing_configuration(self):
        camera = SharedCameraMotion(max_width=160, diff_threshold=30, min_points=20)
        frame = texture()
        camera.apply(frame)
        camera.apply(frame)
        self.assertTrue(camera.last_result.reliable)
        camera.reset_params()
        camera.apply(frame)
        self.assertFalse(camera.last_result.reliable)
        self.assertEqual(camera.calls, 1)
        self.assertEqual((camera.max_width, camera.diff_threshold, camera.min_points),
                         (160, 30, 20))

    def test_staged_full_boxes_override_tracker_subset_once(self):
        frame = texture()
        camera = SharedCameraMotion()
        full_boxes = np.float32([[0, 0, 320, 240]])
        camera.set_detections(full_boxes)
        full_boxes[:] = 0  # Staging owns its copy, not the producer's buffer.
        camera.apply(frame, EMPTY)
        camera.apply(frame, EMPTY)
        self.assertFalse(camera.last_result.reliable)  # Previous full exclusion.
        camera.apply(frame, EMPTY)
        self.assertTrue(camera.last_result.reliable, camera.last_result.reason)
        camera.set_detections([[0, 0, 320, 240]])
        camera.apply(frame, EMPTY)
        self.assertFalse(camera.last_result.reliable)  # Current full exclusion.
        self.assertEqual(camera.calls, 4)

    def test_reset_clears_staged_detections(self):
        frame = texture()
        camera = SharedCameraMotion()
        camera.set_detections([[0, 0, 320, 240]])
        camera.reset_params()
        camera.apply(frame, EMPTY)
        camera.apply(frame, EMPTY)
        self.assertTrue(camera.last_result.reliable, camera.last_result.reason)


class CameraMotionResultTests(unittest.TestCase):
    def result(self):
        valid = np.zeros((6, 8), np.uint8)
        valid[1:5, 2:7] = 255
        motion = np.zeros_like(valid)
        motion[1:3, 2:4] = 255  # Four moving pixels / twenty valid pixels.
        motion[:, :2] = 255  # Invalid pixels must not contribute.
        return CameraMotionResult(np.float32([[0, -1, 7], [1, 0, 4]]),
                                  True, 'ok', 40, 1.0, 0.5, motion, valid)

    def test_transform_point_applies_previous_to_current_affine(self):
        np.testing.assert_allclose(self.result().transform_point((3, 2)), (5, 7))

    def test_motion_ratio_clips_xyxy_and_counts_only_valid_pixels(self):
        result = self.result()
        self.assertAlmostEqual(result.motion_ratio((-10, -10, 20, 20)), 0.2)
        self.assertAlmostEqual(result.motion_ratio((2, 1, 4, 3)), 1.0)
        self.assertAlmostEqual(result.motion_ratio((4, 3, 7, 5)), 0.0)
        for box in ((0, 0, 2, 6), (9, 9, 12, 12), (4, 4, 2, 2),
                    (2, 2, 2, 4), (np.nan, 0, 2, 4), (0, 0, np.inf, 4)):
            with self.subTest(box=box):
                self.assertEqual(result.motion_ratio(box), 0.0)


if __name__ == '__main__':
    unittest.main()
