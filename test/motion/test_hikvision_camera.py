"""Tests for credential loading and URL handling; no camera is contacted."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

APP_ROOT = Path(__file__).resolve().parent.parent.parent
if str(APP_ROOT) not in sys.path:  # Tests live in test/motion/; hikvision_camera.py is at root.
    sys.path.insert(0, str(APP_ROOT))

import hikvision_camera
from hikvision_camera import CameraError, HikvisionClient, _parse_env, load_env

SECRET = 'unit-secret-value'


class ParseEnvTests(unittest.TestCase):
    def test_parses_values_and_ignores_comments_blank_and_malformed_lines(self):
        text = ('# comment\n\nCAMERA_USER = admin \n'
                'CAMERA_PASSWORD="quoted secret"\nNOT_A_PAIR\n=MISSING_KEY\n')
        self.assertEqual(_parse_env(text),
                         {'CAMERA_USER': 'admin', 'CAMERA_PASSWORD': 'quoted secret'})

    def test_missing_file_is_not_an_error(self):
        self.assertFalse(load_env(Path(tempfile.gettempdir()) / 'no-such-hikvision.env'))


class LoadEnvTests(unittest.TestCase):
    """Environment mutation happens in a child process, never in this one."""

    def run_child(self, script):
        result = subprocess.run([sys.executable, '-c', script], cwd=str(APP_ROOT),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def child_preamble(self, temp_env):
        return (f'import sys; sys.path.insert(0, {str(APP_ROOT)!r})\n'
                f'from hikvision_camera import load_env\n'
                f'load_env(r"{temp_env}")\n')

    def test_values_are_loaded_and_real_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / '.env'
            env_file.write_text(f'CAMERA_USER=file-user\n'
                                f'CAMERA_PASSWORD={SECRET}\n'
                                f'PTZ_PASSWORD={SECRET}\n', encoding='utf-8')
            os.environ.pop('CAMERA_USER', None)
            script = (
                'import os\n'
                'os.environ.pop("CAMERA_PASSWORD", None)\n'
                'os.environ["PTZ_PASSWORD"] = "already-set"\n'
                + self.child_preamble(env_file) +
                'print("user", os.environ.get("CAMERA_USER"))\n'
                'print("ptz", os.environ.get("PTZ_PASSWORD"))\n'
                'print("cam", os.environ.get("CAMERA_PASSWORD"))\n')
            output = self.run_child(script)
        self.assertIn('user file-user', output)
        self.assertIn('ptz already-set', output)  # A real env var still wins.
        self.assertIn(f'cam {SECRET}', output)
        self.assertNotIn('CAMERA_USER', os.environ)  # Parent untouched.

    def test_workspace_env_file_resolves_the_admin_account_without_exposing_the_secret(self):
        env_file = APP_ROOT / '.env'
        if not env_file.is_file():
            self.skipTest('no workspace .env present')
        expected = _parse_env(env_file.read_text(encoding='utf-8'))
        script = (self.child_preamble(APP_ROOT / '.env') +
                  'import os\n'
                  'print("user", os.environ.get("CAMERA_USER"))\n'
                  'print("has-password", bool(os.environ.get("CAMERA_PASSWORD")))\n')
        output = self.run_child(script)
        self.assertIn(f'user {expected["CAMERA_USER"]}', output)
        self.assertIn('has-password True', output)
        self.assertNotIn(expected['CAMERA_PASSWORD'], output)


class ClientTests(unittest.TestCase):
    def setUp(self):
        # Isolate from the real workspace .env and from any ambient credentials.
        self.environment = patch.dict(os.environ, {
            'CAMERA_HOST': '127.0.0.1', 'CAMERA_USER': 'admin',
            'CAMERA_PASSWORD': SECRET}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.file_loader = patch('hikvision_camera.load_env', return_value=False)
        self.file_loader.start()
        self.addCleanup(self.file_loader.stop)
        for name in ('PTZ_PASSWORD',):
            os.environ.pop(name, None)

    def test_env_file_credentials_avoid_the_password_prompt(self):
        with patch('hikvision_camera.getpass.getpass',
                   side_effect=AssertionError('must not prompt when a password is configured')):
            client = HikvisionClient(timeout=.2)
        self.assertEqual(client.host, '127.0.0.1')
        self.assertEqual(urlsplit(client.rtsp_source()).username, 'admin')

    def test_special_characters_are_url_encoded_in_the_rtsp_source(self):
        client = HikvisionClient(password='p@ss:word/1')
        source = client.rtsp_source(102)
        parsed = urlsplit(source)
        self.assertEqual(parsed.scheme, 'rtsp')
        self.assertEqual(parsed.hostname, '127.0.0.1')
        self.assertEqual(unquote(parsed.password), 'p@ss:word/1')  # Decodes back.
        self.assertNotIn('p@ss', source)
        self.assertTrue(parsed.path.endswith('/Streaming/Channels/102'))

    def test_missing_password_is_rejected_without_prompting(self):
        os.environ.pop('CAMERA_PASSWORD', None)
        with patch('hikvision_camera.getpass.getpass', return_value=''):
            with self.assertRaisesRegex(ValueError, 'password is required'):
                HikvisionClient()

    def test_connection_failure_message_never_contains_the_password(self):
        client = HikvisionClient(port=1, timeout=.2)
        with self.assertRaises(CameraError) as caught:
            client.device_info()
        self.assertNotIn(SECRET, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
