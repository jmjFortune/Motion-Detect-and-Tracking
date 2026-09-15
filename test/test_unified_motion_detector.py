"""Real dependency integration tests; no camera or network connections."""

import sys
import csv
import io
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:  # Tests live in test/ but import the root modules.
    sys.path.insert(0, str(APP_ROOT))

import cv2
import numpy as np
from ultralytics.engine.results import Boxes

from unified_motion_detector import (
    FrameClock, UnifiedMotionDetector, build_parser, parse_source,
    safe_source_label, write_csv_frame, CSV_FIELDS,
    run,
)


class UnifiedMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = APP_ROOT / 'yolo26n.pt'
        if not cls.model_path.is_file():
            raise AssertionError('Local YOLO weight required for integration tests')
        cls.detector = UnifiedMotionDetector(model_path=str(cls.model_path))
        rng = np.random.default_rng(200)
        cls.background = rng.integers(20, 220, (240, 320, 3), dtype=np.uint8)

    def setUp(self):
        self.detector.reset()

    def boxes(self, x=35, y=55, empty=False):
        data = np.empty((0, 6), np.float32) if empty else np.array(
            [[x, y, x+55, y+95, .9, 0]], np.float32)
        return Boxes(data, (240, 320))

    def test_real_botsort_preserves_id_and_reuses_gmc_once(self):
        frame = self.background.copy()
        frame[55:150, 35:90] = 100
        first = self.detector.update_tracks(self.boxes(), frame)
        self.assertEqual(self.detector.camera.calls, 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].box, (35., 55., 90., 150.))
        next_frame = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 3]]), (320, 240))
        second = self.detector.update_tracks(self.boxes(39, 58), next_frame)
        self.assertEqual(self.detector.camera.calls, 2)
        self.assertTrue(self.detector.camera.last_result.reliable)
        self.assertEqual(second[0].track_id, first[0].track_id)
        self.assertEqual(second[0].box, (39., 58., 94., 153.))
        self.assertIs(self.detector.tracker.gmc, self.detector.camera)
        self.assertIsNone(self.detector.tracker.encoder)

    def test_empty_detections_still_update_shared_gmc(self):
        for _ in range(3):
            observations = self.detector.update_tracks(self.boxes(empty=True), self.background)
            self.assertEqual(observations, [])
        self.assertEqual(self.detector.camera.calls, 3)
        self.assertTrue(self.detector.camera.last_result.reliable)

    def sequence(self, independent_motion):
        states = []
        ids = []
        for i in range(9):
            camera_shift = i*3
            x = 35 + camera_shift + (i*3 if independent_motion else 0)
            frame = cv2.warpAffine(self.background, np.float32(
                [[1, 0, camera_shift], [0, 1, 0]]), (320, 240))
            frame[55:150, x:x+55] = 100
            observations = self.detector.update_tracks(self.boxes(x), frame)
            targets = self.detector.classifier.update(
                observations, self.detector.camera.last_result, i*.1)
            self.assertEqual(len(targets), 1)
            states.append(targets[0].state)
            ids.append(targets[0].observation.track_id)
        self.assertEqual(len(set(ids)), 1)
        return states

    def test_camera_translation_does_not_create_motion(self):
        states = self.sequence(False)
        self.assertNotIn('MOVING', states)
        self.assertEqual(states[-1], 'STATIC')

    def test_camera_and_independent_object_motion(self):
        states = self.sequence(True)
        self.assertEqual(states[-1], 'MOVING')

    def test_camera_follows_object_with_fixed_image_center(self):
        targets = []
        for i in range(9):
            frame = cv2.warpAffine(self.background, np.float32(
                [[1, 0, -i*3], [0, 1, 0]]), (320, 240))
            frame[55:150, 35:90] = 100
            observations = self.detector.update_tracks(self.boxes(), frame)
            targets = self.detector.classifier.update(
                observations, self.detector.camera.last_result, i*.1)
        self.assertEqual(targets[0].state, 'MOVING')
        self.assertGreater(targets[0].speed_px_s, 20.)

    def test_small_rotation_and_scale_preserve_static_object(self):
        base = self.background.copy()
        base[55:150, 35:90] = 100
        corners = np.float32([[[35, 55], [90, 55], [90, 150], [35, 150]]])
        targets = []
        for i in range(9):
            warp = cv2.getRotationMatrix2D((160, 120), i*.3, 1.005**i)
            frame = cv2.warpAffine(base, warp, (320, 240))
            transformed = cv2.transform(corners, warp)[0]
            low = transformed.min(axis=0)
            high = transformed.max(axis=0)
            boxes = Boxes(np.array([[*low, *high, .9, 0]], np.float32), (240, 320))
            observations = self.detector.update_tracks(boxes, frame)
            targets = self.detector.classifier.update(
                observations, self.detector.camera.last_result, i*.1)
            self.assertNotIn('MOVING', [target.state for target in targets])
        self.assertTrue(self.detector.camera.last_result.reliable)
        self.assertEqual(targets[0].state, 'STATIC')

    def test_actual_local_yolo_headless_inference(self):
        for i in range(2):
            result = self.detector.process(np.zeros((240, 320, 3), np.uint8), i*.1)
            self.assertEqual(result.targets, [])
            self.assertEqual(result.selection.status, 'NONE')
            self.assertFalse(result.camera.reliable)
        self.assertEqual(self.detector.camera.calls, 2)

    def test_real_file_runner_timestamps_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'blank.avi'
            output = Path(directory) / 'motion.csv'
            video = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
            self.assertTrue(video.isOpened())
            for _ in range(3):
                video.write(np.zeros((240, 320, 3), np.uint8))
            video.release()
            capture = cv2.VideoCapture(str(source))
            args = build_parser().parse_args([
                '--source', str(source), '--headless', '--max-frames', '3', '--csv', str(output)])
            # Only inject existing resources, never detection outputs: this runs
            # actual file decoding, YOLO, BOTSORT, motion and CSV code.
            with patch('unified_motion_detector.open_capture', return_value=capture), \
                    patch('unified_motion_detector.UnifiedMotionDetector', return_value=self.detector), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            self.assertFalse(capture.isOpened())
            with output.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([float(row['timestamp']) for row in rows], [0., .1, .2])
            self.assertEqual([row['selection_status'] for row in rows], ['NONE'] * 3)

    def test_processing_error_closes_capture_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'one.avi'
            output = Path(directory) / 'motion.csv'
            video = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
            self.assertTrue(video.isOpened())
            video.write(np.zeros((240, 320, 3), np.uint8))
            video.release()
            capture = cv2.VideoCapture(str(source))
            args = build_parser().parse_args(['--source', str(source), '--headless', '--csv', str(output)])
            with patch('unified_motion_detector.open_capture', return_value=capture), \
                    patch('unified_motion_detector.UnifiedMotionDetector', return_value=self.detector), \
                    patch.object(self.detector, 'process', side_effect=RuntimeError('inference failed')):
                with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                    run(args)
            self.assertFalse(capture.isOpened())
            # Windows refuses deletion of open files, exercising CSV finally.
            output.unlink()

    def test_csv_lost_target_has_no_stale_box(self):
        states = self.sequence(True)
        self.assertEqual(states[-1], 'MOVING')
        # Use the last *measured* reliable sample from the sequence, not a new
        # unrelated image/box that would manufacture an extra movement sample.
        targets = self.detector.classifier.update(
            self.detector.update_tracks(self.boxes(89), self.sequence_last_frame()),
            self.detector.camera.last_result, .9)
        selected = self.detector.selector.select(targets, .9, self.background.shape)
        self.assertIsNotNone(selected.track_id)
        lost = self.detector.selector.select([], 1., self.background.shape)
        self.assertEqual(lost.status, 'LOST')
        from unified_motion_detector import FrameResult
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        write_csv_frame(writer, 1., FrameResult([], lost, self.detector.camera.last_result))
        row = next(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(row['selection_status'], 'LOST')
        self.assertEqual(row['selected_id'], str(lost.track_id))
        for name in ('track_id', 'x1', 'y1', 'x2', 'y2'):
            self.assertEqual(row[name], '')

    def sequence_last_frame(self):
        frame = cv2.warpAffine(self.background, np.float32([[1, 0, 27], [0, 1, 0]]), (320, 240))
        frame[55:150, 89:144] = 100
        return frame

    def test_unknown_semantic_class_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            UnifiedMotionDetector(str(self.model_path), classes='not-a-model-class')


class InputTests(unittest.TestCase):
    def test_sources_and_secret_free_label(self):
        self.assertEqual(parse_source('0'), 0)
        self.assertEqual(parse_source('video.mp4'), 'video.mp4')
        label = safe_source_label('rtsp://admin:secret@camera/live')
        self.assertNotIn('admin', label)
        self.assertNotIn('secret', label)
        self.assertNotIn('camera', label)

    def test_source_timestamps_independent_of_inference_time(self):
        clock = FrameClock(is_live=False, source_fps=20.)
        self.assertEqual(clock.timestamp(0, 90.), 0.)
        self.assertEqual(clock.timestamp(2, 150.), .1)
        live = FrameClock(is_live=True, source_fps=0.)
        self.assertEqual(live.timestamp(0, 12.), 0.)
        self.assertAlmostEqual(live.timestamp(1, 12.4), .4)
        with self.assertRaises(ValueError):
            FrameClock(is_live=False, source_fps=0.)

    def test_cli_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.source, '0')
        self.assertEqual(args.imgsz, 416)
        self.assertEqual(args.classes, 'person,dog,cat')
        self.assertFalse(args.show_all)

    def test_unopened_input_is_released(self):
        class ClosedCapture:
            released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        capture = ClosedCapture()
        with patch('unified_motion_detector.open_capture', return_value=capture):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open'):
                run(build_parser().parse_args(['--headless', '--source', 'missing.mp4']))
        self.assertTrue(capture.released)


if __name__ == '__main__':
    unittest.main()
