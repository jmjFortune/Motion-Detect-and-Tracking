import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ball_camera_track import build_parser, check_args, draw_error_curve, run, sharpness


class TrackingApplicationTests(unittest.TestCase):
    def test_default_is_offline_and_never_constructs_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'simulation'
            args = build_parser().parse_args(['--output-dir', str(output)])
            with patch('ball_camera_track.client_from_args', side_effect=AssertionError('network')), \
                    patch('ball_camera_track.HikvisionClient', side_effect=AssertionError('network')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['camera_connections'], 0)
            self.assertEqual(report['ptz_move_requests_sent'], 0)
            self.assertTrue((output/'tracking-error.png').is_file())

    def test_error_curve_is_valid_image(self):
        rows = [dict(time=i*.1, error_x_px=50-i, error_y_px=-20+i*.2) for i in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'curve.png'
            self.assertTrue(draw_error_curve(rows, path))
            image = cv2.imread(str(path))
            self.assertEqual(image.shape, (650, 1200, 3))
            self.assertGreater(image.var(), 0)

    def test_sharpness_prefers_detailed_frame(self):
        plain = np.full((120, 160, 3), 100, np.uint8)
        detailed = plain.copy()
        detailed[:, ::4] = 255
        self.assertGreater(sharpness(detailed), sharpness(plain))

    def test_invalid_or_unbounded_runs_rejected(self):
        for flags in (['--duration', '100'], ['--duration', 'nan'],
                      ['--tracking-seconds', '31'], ['--control-hz', '100'],
                      ['--max-excursion', '1000']):
            with self.assertRaises(ValueError):
                check_args(build_parser().parse_args(flags))

    def test_continuous_mode_is_explicit(self):
        args = build_parser().parse_args(['--continuous'])
        self.assertTrue(args.continuous)


if __name__ == '__main__':
    unittest.main()
