import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from camera_motion import SharedCameraMotion
from ptz_motion_verification import BackgroundMotionVerifier, VerificationFrame, verify_sequence


def texture(seed=73):
    return np.random.default_rng(seed).integers(0, 180, (240, 320, 3), dtype=np.uint8)


class IncrementalEvidenceTests(unittest.TestCase):
    def test_large_total_pan_from_small_adjacent_pairs(self):
        original = texture()
        samples = [VerificationFrame(i*.1, cv2.warpAffine(original,
                    np.float32([[1, 0, i*4], [0, 1, 0]]), (320, 240))) for i in range(25)]
        summary, rows = verify_sequence(samples, 'pan', lambda frame: [])
        self.assertTrue(summary['reliable'], summary)
        self.assertGreater(summary['max_axis_excursion_pixels'], 90.)
        self.assertFalse(rows[0]['reliable'])

    def test_vertical_excursion_uses_vertical_not_horizontal_displacement(self):
        source = texture()
        samples = [VerificationFrame(i*.1, cv2.warpAffine(source,
                    np.float32([[1, 0, 0], [0, 1, i*2]]), (320, 240))) for i in range(15)]
        tilt, _ = verify_sequence(samples, 'tilt', lambda frame: [])
        pan, _ = verify_sequence(samples, 'pan', lambda frame: [])
        self.assertTrue(tilt['reliable'], tilt)
        self.assertFalse(pan['reliable'])

    def test_moving_foreground_does_not_prove_camera_motion(self):
        background = texture()
        object_texture = texture(seed=90)[40:200, 60:240]
        samples = []
        boxes = []
        for i in range(15):
            frame = background.copy()
            x = 50+i*3
            frame[50:210, x:x+180] = object_texture
            boxes.append([[x, 50, x+180, 210]])
            samples.append(VerificationFrame(i*.1, frame))
        iterator = iter(boxes)
        report, _ = verify_sequence(samples, 'pan', lambda frame: next(iterator))
        self.assertFalse(report['reliable'], report)
        self.assertLess(report['max_axis_excursion_pixels'], .5)

    def test_static_and_textureless_video_cannot_verify_motion(self):
        for image in [texture(), np.full((240, 320, 3), 70, np.uint8)]:
            samples = [VerificationFrame(i*.1, image.copy()) for i in range(12)]
            report, _ = verify_sequence(samples, 'pan', lambda frame: [])
            self.assertFalse(report['reliable'], report)

    def test_occlusion_and_time_gaps_break_accumulation(self):
        verifier = BackgroundMotionVerifier()
        source = texture()
        for i, t in enumerate([0., .1, .2, 1., 1.1, 1.2]):
            frame = cv2.warpAffine(source, np.float32([[1, 0, 2*i], [0, 1, 0]]), (320, 240))
            row = verifier.observe(frame, [], t)
            if t == 1.:
                self.assertFalse(row['reliable'])
                self.assertEqual(row['chain_pairs'], 0)
                self.assertEqual(row['reason'], 'timestamp gap')
        self.assertFalse(verifier.summary()['reliable'])
        self.assertTrue(all(run['pairs'] <= 2 for run in verifier.summary()['continuous_runs']))

    def test_duplicate_or_reversed_timestamps_rejected(self):
        verifier = BackgroundMotionVerifier()
        image = texture()
        verifier.observe(image, [], 1.)
        for t in [1., .9, float('nan')]:
            with self.assertRaises(ValueError):
                verifier.observe(image, [], t)

    def test_changing_osd_does_not_prove_camera_motion(self):
        source = texture()
        samples = []
        for i in range(12):
            image = source.copy()
            image[:25] = 0
            cv2.putText(image, str(i)*10, (4+i, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            samples.append(VerificationFrame(i*.1, image))
        report, _ = verify_sequence(samples, 'pan', lambda image: [])
        self.assertFalse(report['reliable'])
        self.assertLess(report['max_axis_excursion_pixels'], .5)

    def test_warps_are_composed_not_translation_summed(self):
        class MockMotion:
            def __init__(self):
                self.index = 0

            def apply(self, frame, boxes):
                warp = np.float32([[0, -1, 100], [1, 0, 0]]) if self.index == 1 else np.float32([[1, 0, 5], [0, 1, 0]])
                self.last_result = SimpleNamespace(warp=warp, reliable=self.index > 0, reason='mock',
                                                   tracked_points=50, inlier_ratio=1., coverage=.5)
                self.index += 1

        verifier = BackgroundMotionVerifier(motion=MockMotion())
        frame = np.zeros((100, 100, 3), np.uint8)
        verifier.observe(frame, [], 0.)
        verifier.observe(frame, [], .1)
        row = verifier.observe(frame, [], .2)
        self.assertAlmostEqual(row['chain_dx'], 5.)
        self.assertAlmostEqual(row['chain_dy'], 0.)

    def test_gmc_quality_defaults_are_preserved(self):
        verifier = BackgroundMotionVerifier()
        self.assertIsInstance(verifier.motion, SharedCameraMotion)
        self.assertEqual(verifier.motion.min_points, 12)


if __name__ == '__main__':
    unittest.main()
