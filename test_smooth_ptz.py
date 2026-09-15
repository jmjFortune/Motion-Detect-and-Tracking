import contextlib
import io
import math
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smooth_ptz import SmoothRoundTrip, SpeedTransition, sample_plan
from ptz_smooth_test import build_parser, check_args, lease_stop_worker, run


class SmoothPlanTests(unittest.TestCase):
    def test_s_curves_are_bounded_monotone_and_end_at_rest(self):
        for start, end in [(0., 24.), (24., -24.), (-24., 0.), (0., 1.), (3., 3.)]:
            curve = SpeedTransition(start, end)
            previous = start
            for index in range(1001):
                speed, acc, jerk = curve.sample(curve.duration*index/1000)
                self.assertLessEqual(abs(acc), 30.+1e-8)
                self.assertLessEqual(abs(jerk), 60.+1e-8)
                self.assertGreaterEqual(speed, min(start, end)-1e-8)
                self.assertLessEqual(speed, max(start, end)+1e-8)
                self.assertGreaterEqual((speed-previous)*(1 if end >= start else -1), -1e-8)
                previous = speed
            self.assertEqual(curve.sample(curve.duration), (end, 0., 0.))

    def test_acceleration_is_continuous_at_s_curve_junctions(self):
        curve = SpeedTransition(0., 24.)
        for boundary in [curve.tj, curve.tj+curve.ta, curve.duration]:
            self.assertAlmostEqual(curve.sample(boundary-1e-7)[1], curve.sample(boundary+1e-7)[1], places=4)

    def test_two_axes_plans_have_software_derivative_limits(self):
        for peak, direction in [(24., -1), (18., 1)]:
            rows = SmoothRoundTrip(peak=peak, first_sign=direction).generate()
            self.assertLessEqual(max(abs(row['speed']) for row in rows), peak)
            self.assertLessEqual(max(abs(row['acceleration']) for row in rows), 30.)
            self.assertLessEqual(max(abs(row['jerk']) for row in rows), 60.)
            for a, b in zip(rows, rows[1:]):
                dt = b['t']-a['t']
                self.assertLessEqual(abs((b['speed']-a['speed'])/dt), 30.+1e-4)
                self.assertLessEqual(abs((b['acceleration']-a['acceleration'])/dt), 60.+1e-4)
            self.assertEqual(round(rows[-1]['speed']), 0)
            area = sum(.5*(a['speed']+b['speed'])*(b['t']-a['t']) for a, b in zip(rows, rows[1:]))
            self.assertLess(abs(area), .1)

    def test_numerical_second_order_filter_matches_exact_step(self):
        profile = SmoothRoundTrip()
        profile.target = lambda t: 1.
        profile.duration = 5.
        # A sustained step is a filter test, not a rest-to-rest movement plan.
        with patch('smooth_ptz.validate_plan'):
            rows = profile.generate()
        row = sample_plan(rows, .7)
        self.assertAlmostEqual(row['speed'], 1-(1+4*.7)*math.exp(-4*.7), places=7)

    def test_invalid_profiles_rejected(self):
        for kwargs in [{'peak': 100}, {'peak': float('nan')}, {'omega': 0}, {'first_sign': 0}]:
            with self.assertRaises(ValueError):
                SmoothRoundTrip(**kwargs)
        with self.assertRaises(ValueError):
            SpeedTransition(0, 1, max_jerk=0)

    def test_longer_cruise_increases_excursion_without_raising_derivative_limits(self):
        short = SmoothRoundTrip().generate()
        longer = SmoothRoundTrip(cruise_seconds=2.).generate()
        def negative_area(rows):
            return sum(max(0., -.5*(a['speed']+b['speed']))*(b['t']-a['t'])
                       for a, b in zip(rows, rows[1:]))
        self.assertGreater(negative_area(longer), 2*negative_area(short))
        self.assertAlmostEqual(longer[-1]['t']-short[-1]['t'], 3.4)
        self.assertLessEqual(max(abs(row['speed']) for row in longer), 24.)
        self.assertLessEqual(max(abs(row['acceleration']) for row in longer), 30.)
        self.assertLessEqual(max(abs(row['jerk']) for row in longer), 60.)
        self.assertEqual(round(longer[-1]['speed']), 0)


class SmoothScriptSafetyTests(unittest.TestCase):
    def test_offline_never_connects(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--output-dir', str(Path(directory)/'plan')])
            with patch('ptz_smooth_test.client_from_args', side_effect=AssertionError('network')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)

    def test_tilt_only_plan_does_not_include_pan(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'tilt-only'
            args = build_parser().parse_args(['--axis', 'tilt', '--output-dir', str(output)])
            with patch('ptz_smooth_test.client_from_args', side_effect=AssertionError('network')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(set(report['axes']), {'tilt'})
            self.assertFalse((output/'pan-planned.csv').exists())

    def test_unsafe_parameters_rejected(self):
        for flags in (['--pan-speed', '100'], ['--tilt-speed', '30'], ['--hz', '100'],
                      ['--omega', 'nan'], ['--cruise', '100'], ['--cruise', 'nan'], ['--view']):
            with self.assertRaises(ValueError):
                check_args(build_parser().parse_args(flags))

    def test_watchdog_stale_heartbeat_trips_and_repeatedly_stops(self):
        class Event:
            def __init__(self, set=False):
                self.value = set

            def is_set(self):
                return self.value

            def set(self):
                self.value = True

            def wait(self, timeout):
                return self.value

        from hikvision_camera import CameraError
        device = SimpleNamespace(ptz_status=lambda: {}, stop=unittest.mock.Mock(
            side_effect=[CameraError('lost reply'), None, None, None]))
        heartbeat = SimpleNamespace(value=0., get_lock=contextlib.nullcontext)
        ready, armed, done, tripped = Event(), Event(True), Event(), Event()
        results = queue.Queue()
        with patch('ptz_smooth_test.HikvisionClient', return_value=device), \
                patch('ptz_smooth_test.time.sleep'):
            lease_stop_worker('192.168.1.64', 'admin', 'fake', 80, ready, armed, done,
                              tripped, heartbeat, results, 10.)
        self.assertTrue(ready.is_set())
        self.assertTrue(tripped.is_set())
        self.assertEqual(device.stop.call_count, 4)
        self.assertEqual([results.get_nowait() for _ in range(4)], [False, True, True, True])

    def test_verification_runs_after_stop_and_success_allows_next_axis(self):
        import numpy as np
        import json
        from unittest.mock import Mock
        class Clock:
            value = 100.

            def __call__(self):
                self.value += .025
                return self.value

        clock = Clock()
        class Event:
            value = False

            def set(self):
                self.value = True

            def is_set(self):
                return self.value

            def wait(self, timeout):
                return self.value

        class Process:
            def __init__(self, target, args):
                self.args, self.alive = args, False

            def start(self):
                self.alive = True
                self.args[4].set()  # Stop watchdog ready.
                for _ in range(4):
                    self.args[9].put(True)

            def join(self, timeout):
                self.alive = False

            def is_alive(self):
                return self.alive

        context = SimpleNamespace(Event=Event, Process=Process, Queue=queue.Queue,
                                  Value=lambda *args: SimpleNamespace(value=0., get_lock=contextlib.nullcontext))
        active = [False]
        def stop():
            active[0] = False
        device = SimpleNamespace(host='192.168.1.64', _username='admin', _password='fake',
            ptz_status=lambda: {'azimuth': '2952', 'elevation': '155', 'absoluteZoom': '10'},
            ptz_capabilities=lambda: {'limits_raw': {'azimuth': (0, 3600), 'elevation': (-900, 2700)}},
            rtsp_source=lambda: 'fake-no-network', stop=Mock(side_effect=stop))
        image = np.zeros((240, 320, 3), np.uint8)
        class Capture:
            sequence = 0

            def get(self, *args, **kwargs):
                self.sequence += 1
                return self.sequence, clock(), image.copy()

            def close(self):
                pass
        def move(client, pan, tilt):
            active[0] = bool(pan or tilt)
        verified_axes = []
        def verify(samples, axis, detect, progress):
            self.assertFalse(active[0], 'Semantic inference must not block an active control loop')
            self.assertGreater(device.stop.call_count, 0)
            self.assertGreater(len(samples), 1)
            self.assertTrue(all(b.timestamp > a.timestamp for a, b in zip(samples, samples[1:])))
            self.assertTrue(all(not frame.frame.any() for frame in samples), 'Evidence must be raw, not labelled')
            verified_axes.append(axis)
            return {'reliable': True, 'reason': 'mock verified', 'max_axis_excursion_pixels': 20.}, \
                [{'timestamp': sample.timestamp, 'reliable': True} for sample in samples]
        plan = [dict(t=0., target=0., speed=0., acceleration=0., jerk=0.),
                dict(t=.2, target=10., speed=10., acceleration=0., jerk=0.),
                dict(t=.8, target=0., speed=0., acceleration=0., jerk=0.)]
        video = Mock()
        video.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--execute', '--output-dir', str(Path(directory)/'test')])
            with patch('ptz_smooth_test.client_from_args', return_value=device), \
                    patch('ptz_smooth_test.create_foreground_detector', return_value=lambda image: []), \
                    patch('ptz_smooth_test.verify_sequence', side_effect=verify), \
                    patch('ptz_smooth_test.continuous_move', side_effect=move), \
                    patch('ptz_smooth_test.mp.get_context', return_value=context), \
                    patch('ptz_smooth_test.SmoothRoundTrip.generate', return_value=plan), \
                    patch('ptz_smooth_test.time.monotonic', side_effect=clock), \
                    patch('ptz_smooth_test.time.sleep'), \
                    patch('ball_camera_detect.LatestFrameCapture', return_value=Capture()), \
                    patch('cv2.VideoWriter', return_value=video), \
                    patch('cv2.imwrite', return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)
            report = json.loads((Path(directory)/'test'/'report.json').read_text())
            self.assertIsNone(report['failure'])
            self.assertTrue(report['feedback_stationary_after_stop'])
        self.assertEqual(verified_axes, ['pan', 'tilt'])


if __name__ == '__main__':
    unittest.main()
