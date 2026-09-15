import contextlib
import io
import math
import tempfile
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from hikvision_camera import CameraError, HikvisionClient
from ptz_control import AxisController, SecondOrderReference, VelocityRamp
from ptz_second_order_test import build_parser, check_args, run


class ControllerTests(unittest.TestCase):
    def test_reference_matches_exact_critical_step_response(self):
        reference = SecondOrderReference(omega=1.5)
        position, velocity = reference.update(3., .7)
        self.assertAlmostEqual(position, 3.*(1.-(1.+1.5*.7)*math.exp(-1.5*.7)))
        self.assertAlmostEqual(velocity, 3.*1.5**2*.7*math.exp(-1.5*.7))

    def test_variable_dt_matches_one_update(self):
        single = SecondOrderReference()
        repeated = SecondOrderReference()
        single.update(3., 1.)
        for dt in (.13, .21, .17, .49):
            repeated.update(3., dt)
        self.assertAlmostEqual(single.position, repeated.position)
        self.assertAlmostEqual(single.velocity, repeated.velocity)

    def test_target_reversal_preserves_state_and_remains_finite(self):
        reference = SecondOrderReference()
        reference.update(3., 2.)
        position, velocity = reference.position, reference.velocity
        reference.update(0., 1e-6)
        self.assertAlmostEqual(reference.position, position, places=4)
        self.assertAlmostEqual(reference.velocity, velocity, places=4)

    def test_speed_ramp_respects_velocity_and_acceleration(self):
        ramp = VelocityRamp(max_speed=2., max_acceleration=1.)
        for desired in [50.]*80+[-50.]*160:
            speed, acceleration = ramp.update(desired, .07)
            self.assertLessEqual(abs(speed), 2.+1e-9)
            self.assertLessEqual(abs(acceleration), 1.+1e-9)

    def test_axis_controller_caps_normalized_units_and_has_no_integral_windup(self):
        controller = AxisController(gain=10., max_speed=3.)
        for _ in range(80):
            row = controller.update(4., 0., .1)
            self.assertLessEqual(abs(row['command']), 6)
            self.assertLessEqual(abs(row['command_speed']), .6+1e-9)
            self.assertLessEqual(abs(row['command_acceleration']), 1.+1e-9)
        self.assertNotIn('integral', controller.__dict__)

    def test_invalid_input_is_rejected(self):
        for dt in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                SecondOrderReference().update(1., dt)
        with self.assertRaises(ValueError):
            VelocityRamp().update(float('nan'), .1)
        with self.assertRaises(ValueError):
            AxisController(gain=0.)
        with self.assertRaises(ValueError):
            AxisController(max_command=100)


class CameraClientTests(unittest.TestCase):
    def setUp(self):
        self.client = HikvisionClient(password='not-a-real-password')

    def test_pulse_has_timeout_and_zero_zoom(self):
        captured = []
        with patch.object(self.client, 'request_xml', side_effect=lambda *args: captured.append(args)):
            self.client.move_pulse(2, -1)
            self.client.stop()
        path, method, body = captured[0]
        self.assertTrue(path.endswith('/momentary'))
        self.assertEqual(method, 'PUT')
        self.assertEqual(body.find('zoom').text, '0')
        self.assertEqual(body.find('Momentary/duration').text, '300')
        self.assertEqual([child.text for child in captured[1][2]], ['0', '0', '0'])

    def test_invalid_motion_is_rejected_without_network(self):
        for command in (11, -11, 2.5, True):
            with self.assertRaises(ValueError):
                self.client.move_pulse(command, 0)
        with self.assertRaises(ValueError):
            self.client.move_pulse(0, 0, duration_ms=5000)

    def test_http_200_application_rejection_is_not_success(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, *args):
                return b'<ResponseStatus><statusCode>4</statusCode></ResponseStatus>'

        with patch.object(self.client._opener, 'open', return_value=Response()):
            with self.assertRaisesRegex(CameraError, 'ISAPI status 4'):
                self.client.request_xml('/ISAPI/PTZCtrl/channels/1/momentary', 'PUT')


class ScriptSafetyTests(unittest.TestCase):
    def test_simulation_never_constructs_or_moves_camera_client(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--output-dir', str(Path(directory)/'simulation')])
            with patch('ptz_second_order_test.client_from_args', side_effect=AssertionError('network forbidden')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 0)

    def test_failed_motion_reply_still_attempts_stop(self):
        class Device:
            stopped = False

            def ptz_status(self):
                return {'azimuth': '2854', 'elevation': '129', 'absoluteZoom': '10'}

            def ptz_capabilities(self):
                return {'limits_raw': {'azimuth': (0., 3500.), 'elevation': (0., 900.)}}

            def move_pulse(self, *args):
                raise CameraError('reply lost')

            def stop(self):
                self.stopped = True

        device = Device()
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(['--execute', '--output-dir', str(Path(directory)/'physical')])
            with patch('ptz_second_order_test.client_from_args', return_value=device), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args), 1)
        self.assertTrue(device.stopped)

    def test_oversized_or_nonfinite_trials_rejected(self):
        for flags in (['--pan-step', '50'], ['--duration', '1000'], ['--hz', 'nan'],
                      ['--max-speed', '20'], ['--probe', '--execute'], ['--view']):
            with self.assertRaises(ValueError):
                check_args(build_parser().parse_args(flags))


if __name__ == '__main__':
    unittest.main()
