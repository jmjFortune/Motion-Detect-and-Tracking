"""Bounded real-camera test of the camera web UI's continuous PTZ interface.

Default is a dry run. A separate spawned process sends repeated stops even if
the video/main loop stalls. This is NOT device-side expiry or calibrated control.
"""

import argparse
import json
import multiprocessing as mp
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from hikvision_camera import CameraError, HikvisionClient, add_camera_arguments, client_from_args


def continuous_move(client, pan, tilt):
    if not all(type(value) is int and abs(value) <= 60 for value in (pan, tilt)):
        raise ValueError('Web test commands must be integers in [-60,60]')
    if pan and tilt:
        raise ValueError('Test one axis at a time')
    body = ET.Element('PTZData')
    for name, value in [('pan', pan), ('tilt', tilt), ('zoom', 0)]:
        ET.SubElement(body, name).text = str(value)
    return client.request_xml('/ISAPI/PTZCtrl/channels/1/continuous', 'PUT', body)


def stop_worker(host, username, password, port, ready, armed, results, duration):
    """Independent repeated stop attempts; credentials never enter argv/files."""
    try:
        client = HikvisionClient(host, username, password, port, timeout=.8)
        client.ptz_status()  # Warm Digest authentication before allowing motion.
        ready.set()
        if not armed.wait(10.):
            return
        deadline = time.monotonic() + duration
        for offset in (0., .5, 1.):
            time.sleep(max(0., deadline + offset - time.monotonic()))
            try:
                client.stop()
                results.put(True)
            except CameraError:
                results.put(False)
    except Exception:
        results.put(False)


def check_args(args):
    if type(args.speed) is not int or not 15 <= args.speed <= 60:
        raise ValueError('Speed must be 15..60')
    if not .2 <= args.seconds <= .6:
        raise ValueError('Each movement must be .2.. .6 seconds')
    if args.view and not args.execute:
        raise ValueError('--view requires --execute')


def image_shift(before, after):
    import cv2
    import numpy as np
    a = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(a, 600, .015, 12)
    if points is None:
        return dict(reliable=False)
    target, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(31, 31), maxLevel=4)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(b, a, target, None, winSize=(31, 31), maxLevel=4)
    good = (status.ravel() == 1) & (reverse.ravel() == 1)
    good &= np.linalg.norm(back.reshape(-1, 2)-points.reshape(-1, 2), axis=1) < 1.5
    source, dest = points.reshape(-1, 2)[good], target.reshape(-1, 2)[good]
    if len(source) < 20:
        return dict(reliable=False, tracked=len(source))
    warp, mask = cv2.estimateAffinePartial2D(source, dest, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if warp is None or mask is None:
        return dict(reliable=False, tracked=len(source))
    kept = source[mask.ravel() == 1]
    h, w = a.shape
    coverage = float(cv2.contourArea(cv2.convexHull(kept.astype('float32')))/(w*h))
    center = np.array([w/2, h/2, 1.])
    shift = warp @ center - center[:2]
    return dict(reliable=len(kept) >= 20 and coverage > .2,
                inliers=len(kept), coverage=coverage,
                center_shift_pixels=shift.tolist(), magnitude_pixels=float(np.linalg.norm(shift)))


def run(args):
    check_args(args)
    sequence = [('LEFT', -args.speed, 0), ('RIGHT', args.speed, 0)]
    if args.axis == 'both':
        sequence += [('UP', 0, args.speed), ('DOWN', 0, -args.speed)]
    if not args.execute:
        print('DRY RUN: no camera connection or motion.', sequence)
        return 0
    import cv2
    from ball_camera_detect import LatestFrameCapture
    client = client_from_args(args, timeout=.8)
    ctx = mp.get_context('spawn')
    directory = args.output_dir or Path('output') / datetime.now().strftime('ptz-web-%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    initial = client.ptz_status()
    limits = client.ptz_capabilities()['limits_raw']
    capture = writer = guard = None
    attempted = False
    report = dict(initial_status=initial, speed=args.speed, seconds=args.seconds,
                  segments=[], failure=None, stop_accepted=False,
                  device_side_expiry=False, physical_gain_calibrated=False)
    frame_sequence = 0
    window = 'REAL PTZ web-interface test - Q/Esc stops'
    def sample(label, seconds):
        nonlocal frame_sequence, writer
        end = time.monotonic() + seconds
        last = None
        while last is None or time.monotonic() < end:
            frame_sequence, timestamp, frame = capture.get(frame_sequence, timeout=1.)
            if time.monotonic()-timestamp > .5:
                raise CameraError('Stale video; no further motion allowed')
            last = frame
            display = frame.copy()
            cv2.putText(display, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(directory/'physical-test.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), 10.,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise CameraError('Cannot create evidence video')
            writer.write(display)
            if args.view:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 255 in (27, ord('q'), ord('Q')):
                    raise CameraError('User requested stop')
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise CameraError('Viewer closed; stopping test')
            time.sleep(.06)
        return last
    try:
        capture = LatestFrameCapture(client.rtsp_source())
        if args.view:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print('Starting real test in 3 seconds; watch camera/webpage.', flush=True)
        sample('READY - no movement', 3.)
        for label, pan, tilt in sequence:
            status = client.ptz_status()
            for key in ('azimuth', 'elevation'):
                lo, hi = limits[key]
                if not lo+50 < float(status[key]) < hi-50:
                    raise CameraError('Feedback is near a reported limit; stopping test')
            if status['absoluteZoom'] != initial['absoluteZoom']:
                raise CameraError('Zoom changed externally; stopping test')
            print(f'NEXT: {label}; command=({pan},{tilt}); {args.seconds}s', flush=True)
            before = sample('NEXT: '+label, 2.)
            cv2.imwrite(str(directory/(label.lower()+'-before.jpg')), before)
            ready, armed, results = ctx.Event(), ctx.Event(), ctx.Queue()
            guard = ctx.Process(target=stop_worker, args=(client.host, client._username,
                                client._password, args.http_port, ready, armed, results, args.seconds))
            guard.start()
            if not ready.wait(6.):
                raise CameraError('Independent stop protection not ready; no motion sent')
            attempted = True
            armed.set()
            started = time.monotonic()
            try:
                continuous_move(client, pan, tilt)
                sample(label+' - MOVING', max(.01, args.seconds-(time.monotonic()-started)))
            finally:
                client.stop()
                report['stop_accepted'] = True
            guard.join(4.)
            if guard.is_alive():
                raise CameraError('Stop worker did not finish; no further motion')
            acknowledgements = []
            while not results.empty():
                acknowledgements.append(results.get())
            if not any(acknowledgements):
                raise CameraError('Independent stops not acknowledged; no further motion')
            after = sample(label+' - STOPPED', 1.)
            cv2.imwrite(str(directory/(label.lower()+'-after.jpg')), after)
            feedback = client.ptz_status()
            shift = image_shift(before, after)
            segment = dict(direction=label, before=status, after=feedback, image_motion=shift,
                           independent_stop_acknowledgements=acknowledgements)
            report['segments'].append(segment)
            print(json.dumps(segment), flush=True)
            if not shift['reliable'] or shift.get('magnitude_pixels', 0.) < 3.:
                raise CameraError('No reliable visible scene movement; stopping instead of blindly continuing')
            if any(abs(float(feedback[key])-float(initial[key])) > 300
                   for key in ('azimuth', 'elevation')):
                raise CameraError('Feedback excursion exceeded test bound; stopping')
        sample('DONE - STOPPED', 2.)
    except CameraError as error:
        report['failure'] = str(error)
    except KeyboardInterrupt:
        report['failure'] = 'Interrupted by user'
    except Exception as error:
        report['failure'] = type(error).__name__+' during physical test (details omitted)'
    finally:
        if attempted:
            for _ in range(3):
                try:
                    client.stop()
                    report['stop_accepted'] = True
                except CameraError:
                    report['stop_accepted'] = False
                time.sleep(.15)
        if guard is not None:
            guard.join(4.)
        if capture is not None:
            try:
                capture.close()
            except CameraError:
                report['failure'] = report['failure'] or 'Video cleanup failed'
        if writer is not None:
            writer.release()
        if args.view:
            cv2.destroyAllWindows()
    try:
        report['final_status'] = client.ptz_status()
        time.sleep(.5)
        report['feedback_stationary_after_stop'] = client.ptz_status() == report['final_status']
    except CameraError:
        report['failure'] = report['failure'] or 'Final status unavailable'
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('Results:', directory.resolve(), flush=True)
    return int(bool(report['failure']) or not report['stop_accepted'])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--speed', type=int, default=30)
    parser.add_argument('--seconds', type=float, default=.45)
    parser.add_argument('--axis', choices=('pan', 'both'), default='both')
    parser.add_argument('--output-dir', type=Path)
    return parser


if __name__ == '__main__':
    mp.freeze_support()
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (ValueError, CameraError) as error:
        print('PTZ web test failed:', error)
        raise SystemExit(1)
