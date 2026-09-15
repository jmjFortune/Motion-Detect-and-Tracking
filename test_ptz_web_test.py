import contextlib
import io
import unittest
from unittest.mock import patch

from hikvision_camera import HikvisionClient
from ptz_web_test import build_parser, check_args, continuous_move, run


class WebTestSafety(unittest.TestCase):
    def test_dry_run_never_connects(self):
        with patch('ptz_web_test.client_from_args', side_effect=AssertionError('network')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(build_parser().parse_args([])), 0)

    def test_invalid_parameters_rejected(self):
        for flags in (['--speed', '100'], ['--speed', '0'], ['--seconds', '2'],
                      ['--seconds', 'nan'], ['--view']):
            with self.assertRaises(ValueError):
                check_args(build_parser().parse_args(flags))

    def test_request_matches_web_path_and_has_zero_zoom(self):
        client = HikvisionClient(password='fake-test-password')
        with patch.object(client, 'request_xml') as request:
            continuous_move(client, -30, 0)
        path, method, body = request.call_args.args
        self.assertEqual(path, '/ISAPI/PTZCtrl/channels/1/continuous')
        self.assertEqual(method, 'PUT')
        self.assertEqual(body.tag, 'PTZData')
        self.assertEqual([child.text for child in body], ['-30', '0', '0'])

    def test_excessive_or_diagonal_command_never_sent(self):
        client = HikvisionClient(password='fake-test-password')
        with patch.object(client, 'request_xml', side_effect=AssertionError('network')):
            for pan, tilt in [(100, 0), (True, 0), (30, 30), (0, -61)]:
                with self.assertRaises(ValueError):
                    continuous_move(client, pan, tilt)


if __name__ == '__main__':
    unittest.main()
