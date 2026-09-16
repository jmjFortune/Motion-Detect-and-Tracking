"""Small credential-safe ISAPI client.

Credentials resolve in this order: explicit arguments, then ``PTZ_PASSWORD`` /
``CAMERA_PASSWORD`` / ``CAMERA_USER`` / ``CAMERA_HOST`` from the process
environment or a local ``.env`` file, then a hidden prompt. Values are never
printed, returned or included in error messages.
"""

import getpass
import ipaddress
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


_ENV_FILE = Path(__file__).resolve().parent / '.env'


def _parse_env(text):
    """Parse simple KEY=VALUE lines; ignore comments and malformed lines."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def load_env(path=None):
    """Load camera credentials from ``.env`` without overriding real environment.

    Returns True when the file was read. Values are never logged or returned, so
    an error path can never print the password.
    """
    target = _ENV_FILE if path is None else Path(path)
    try:
        text = target.read_text(encoding='utf-8')
    except OSError:
        return False
    for key, value in _parse_env(text).items():
        if key and value and key not in os.environ:
            os.environ[key] = value
    return True


class CameraError(RuntimeError):
    pass


def xml_text(root, name, default=''):
    node = root.find('.//{*}' + name)
    return default if node is None or node.text is None else node.text


class HikvisionClient:
    def __init__(self, host=None, username=None, password=None,
                 port=80, timeout=2.0):
        load_env()
        host = host or os.environ.get('CAMERA_HOST') or '192.168.1.64'
        self.host = str(ipaddress.IPv4Address(host))
        if not 1 <= port <= 65535 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Invalid HTTP port or timeout')
        if username is None:
            username = os.environ.get('CAMERA_USER') or 'admin'
        if password is None:
            password = os.environ.get('PTZ_PASSWORD') or os.environ.get('CAMERA_PASSWORD')
        if password is None:
            password = getpass.getpass('Camera password: ')
        if not password:
            raise ValueError('Camera password is required')
        self._username = username
        self._password = password
        self._base = f'http://{self.host}:{port}'
        self.timeout = timeout
        manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        manager.add_password(None, self._base, username, password)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPDigestAuthHandler(manager))

    def request_xml(self, path, method='GET', body=None):
        if not path.startswith('/ISAPI/') or method not in ('GET', 'PUT'):
            raise ValueError('Only local ISAPI GET/PUT requests are supported')
        payload = None if body is None else ET.tostring(body, encoding='utf-8', xml_declaration=True)
        request = urllib.request.Request(self._base + path, data=payload, method=method,
                                         headers={'Content-Type': 'application/xml'})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                root = ET.fromstring(response.read(1024*1024))
        except urllib.error.HTTPError as error:
            raise CameraError(f'Camera returned HTTP {error.code} for {method} {path}') from None
        except (urllib.error.URLError, TimeoutError, OSError, ET.ParseError):
            raise CameraError(f'Camera connection or XML response failed for {method} {path}') from None
        # ISAPI can return an application rejection despite HTTP 200.
        if root.tag.rsplit('}', 1)[-1] == 'ResponseStatus':
            code = xml_text(root, 'statusCode')
            if code != '1':
                raise CameraError(f'Camera rejected {method} {path} (ISAPI status {code})')
        return root

    def device_info(self):
        root = self.request_xml('/ISAPI/System/deviceInfo')
        return {name: xml_text(root, name) for name in
                ('deviceName', 'model', 'firmwareVersion', 'firmwareReleasedDate', 'deviceType')}

    def ptz_status(self, channel=1):
        root = self.request_xml(f'/ISAPI/PTZCtrl/channels/{channel}/status')
        values = {name: xml_text(root, name, None) for name in ('azimuth', 'elevation', 'absoluteZoom')}
        return values

    def ptz_capabilities(self, channel=1):
        root = self.request_xml(f'/ISAPI/PTZCtrl/channels/{channel}/capabilities')
        limits = {}
        for name in ('azimuth', 'elevation'):
            pairs = [(float(node.attrib['min']), float(node.attrib['max']))
                     for node in root.findall('.//{*}' + name)
                     if 'min' in node.attrib and 'max' in node.attrib]
            if pairs:
                limits[name] = (max(pair[0] for pair in pairs), min(pair[1] for pair in pairs))
        return dict(limits_raw=limits, support={name: xml_text(root, name, None)
                     for name in ('panSupport', 'tiltSupport', 'zoomSupport')})

    def move_pulse(self, pan, tilt, channel=1, duration_ms=300):
        if not all(isinstance(v, int) and not isinstance(v, bool) and abs(v) <= 10
                   for v in (pan, tilt)):
            raise ValueError('Conservative test speed commands must be integers in [-10,10]')
        if not 1 <= duration_ms <= 350:
            raise ValueError('Movement pulse duration must be 1..350ms')
        body = ET.Element('PTZData', version='2.0', xmlns='http://www.hikvision.com/ver20/XMLSchema')
        for name, value in (('pan', pan), ('tilt', tilt), ('zoom', 0)):
            ET.SubElement(body, name).text = str(value)
        ET.SubElement(ET.SubElement(body, 'Momentary'), 'duration').text = str(duration_ms)
        # Timed device-side movement; NEVER silently fall back to continuous.
        return self.request_xml(f'/ISAPI/PTZCtrl/channels/{channel}/momentary', 'PUT', body)

    def stop(self, channel=1):
        body = ET.Element('PTZData', version='2.0', xmlns='http://www.hikvision.com/ver20/XMLSchema')
        for name in ('pan', 'tilt', 'zoom'):
            ET.SubElement(body, name).text = '0'
        return self.request_xml(f'/ISAPI/PTZCtrl/channels/{channel}/continuous', 'PUT', body)

    def continuous_move(self, pan, tilt, channel=1, command_limit=30):
        """Set bounded Pan/Tilt velocity; zoom is always explicitly zero."""
        if (type(command_limit) is not int or not 1 <= command_limit <= 60 or
                not all(type(value) is int and abs(value) <= command_limit
                        for value in (pan, tilt))):
            raise ValueError('PTZ velocity command exceeds configured safe limit')
        # Match the camera web UI's proven continuous-drive payload exactly.
        # This model acknowledges the namespaced variant but does not reliably
        # energize the motors for it.
        body = ET.Element('PTZData')
        for name, value in (('pan', pan), ('tilt', tilt), ('zoom', 0)):
            ET.SubElement(body, name).text = str(value)
        return self.request_xml(f'/ISAPI/PTZCtrl/channels/{channel}/continuous', 'PUT', body)

    def rtsp_source(self, channel=102):
        if not isinstance(channel, int) or channel <= 0:
            raise ValueError('Streaming channel must be positive')
        user = urllib.parse.quote(self._username, safe='')
        password = urllib.parse.quote(self._password, safe='')
        return f'rtsp://{user}:{password}@{self.host}:554/Streaming/Channels/{channel}'


def add_camera_arguments(parser):
    parser.add_argument('--host', default='192.168.1.64')
    parser.add_argument('--user', default='admin')
    parser.add_argument('--http-port', type=int, default=80)


def client_from_args(args, timeout=2.0):
    return HikvisionClient(args.host, args.user, port=args.http_port, timeout=timeout)
