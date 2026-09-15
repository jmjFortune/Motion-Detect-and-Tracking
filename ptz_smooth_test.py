"""Continuous PTZ speed demo with S curves and second-order speed-target smoothing.

Default offline only. Explicit --execute runs separate horizontal/vertical
round trips with a separate-process heartbeat watchdog and raw feedback bounds.
No angle accuracy or physical jerk guarantee is asserted.
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import time
from datetime import datetime
from pathlib import Path

from hikvision_camera import CameraError, HikvisionClient, add_camera_arguments, client_from_args
from ptz_web_test import continuous_move
from ptz_motion_verification import VerificationFrame, create_foreground_detector, verify_sequence
from smooth_ptz import SmoothRoundTrip, sample_plan


def lease_stop_worker(host, user, password, port, ready, armed, done, tripped,
                      heartbeat, results, hard_duration, lease=.65):
    try:
        client = HikvisionClient(host, user, password, port, timeout=.8)
        client.ptz_status()
        ready.set()
        if not armed.wait(10.):
            return
        start = time.monotonic()
        while not done.is_set():
            with heartbeat.get_lock():
                last = heartbeat.value
            now = time.monotonic()
            if now-last > lease or now-start > hard_duration:
                tripped.set()
                break
            time.sleep(.03)
        # Continue stopping long enough to cover a possibly outstanding PUT.
        for _ in range(4):
            try:
                client.stop()
                results.put(True)
            except CameraError:
                results.put(False)
            time.sleep(.3)
    except Exception:
        tripped.set()
        results.put(False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--axis', choices=('pan', 'tilt', 'both'), default='both')
    parser.add_argument('--pan-speed', type=float, default=24.)
    parser.add_argument('--tilt-speed', type=float, default=18.)
    parser.add_argument('--omega', type=float, default=4.)
    parser.add_argument('--hz', type=float, default=12.)
    parser.add_argument('--cruise', type=float, default=.3,
                        help='Constant target speed hold per direction, .3..3 seconds')
    parser.add_argument('--verification-model', default='yolo26n.pt',
                        help='Existing local YOLO weights for foreground exclusion')
    parser.add_argument('--output-dir', type=Path)
    return parser


def check_args(args):
    if not all(math.isfinite(v) for v in (args.pan_speed, args.tilt_speed, args.omega, args.hz, args.cruise)):
        raise ValueError('Parameters must be finite')
    if not 15 <= args.pan_speed <= 30 or not 15 <= args.tilt_speed <= 24:
        raise ValueError('Pan speed must be 15..30; tilt speed 15..24')
    if not 2 <= args.omega <= 5 or not 8 <= args.hz <= 15:
        raise ValueError('Omega must be 2..5; command rate 8..15Hz')
    if not .3 <= args.cruise <= 3.:
        raise ValueError('Cruise hold must be .3..3 seconds')
    if args.view and not args.execute:
        raise ValueError('--view requires --execute')


def run(args):
    check_args(args)
    profiles = {}
    if args.axis in ('pan', 'both'):
        profiles['pan'] = SmoothRoundTrip(args.pan_speed, args.omega, first_sign=-1,
                                         cruise_seconds=args.cruise)
    if args.axis in ('tilt', 'both'):
        profiles['tilt'] = SmoothRoundTrip(args.tilt_speed, args.omega, first_sign=1,
                                         cruise_seconds=args.cruise)
    plans = {axis: profile.generate() for axis, profile in profiles.items()}
    directory = args.output_dir or Path('output') / datetime.now().strftime('ptz-smooth-%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    for axis, rows in plans.items():
        with (directory/(axis+'-planned.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    report = dict(mode='physical' if args.execute else 'offline', omega=args.omega, cruise=args.cruise,
                  command_hz=args.hz, axes={}, failure=None,
                  duration_meaning='full round trip including acceleration, reversal and near-zero settling',
                  device_side_expiry=False, physical_gain_calibrated=False,
                  physical_jerk_guaranteed=False, integer_command_jerk_guaranteed=False)
    for axis, rows in plans.items():
        report['axes'][axis] = dict(duration=rows[-1]['t'],
            planned_peak_speed=max(abs(row['speed']) for row in rows),
            planned_peak_acceleration=max(abs(row['acceleration']) for row in rows),
            planned_peak_jerk=max(abs(row['jerk']) for row in rows))
    print('Planned float-command limits (NOT physical motor limits):', json.dumps(report['axes']), flush=True)
    if not args.execute:
        (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('OFFLINE ONLY: no camera connection. Results:', directory.resolve())
        return 0
    import cv2
    from ball_camera_detect import LatestFrameCapture
    detect_foreground = create_foreground_detector(args.verification_model)
    ctx = mp.get_context('spawn')
    client = client_from_args(args, timeout=.5)
    capture = video = guard = done = None
    attempted = False
    last_sequence = 0
    latest_timestamp = None
    evidence_samples = None
    window = 'REAL PTZ SMOOTH - Q/Esc emergency stop'
    def frame(label):
        nonlocal last_sequence, video, latest_timestamp
        # Latest-only video: never wait for a new frame inside the command loop.
        sequence, acquired, raw = capture.get(0, timeout=.1)
        if time.monotonic()-acquired > .5:
            raise CameraError('Stale/failed video; emergency stop')
        latest_timestamp = acquired
        display = raw.copy()
        cv2.putText(display, label, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
        if video is None:
            video = cv2.VideoWriter(str(directory/'physical-test.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                    args.hz, (raw.shape[1], raw.shape[0]))
            if not video.isOpened():
                raise CameraError('Cannot create evidence video')
        if sequence != last_sequence:
            video.write(display)
            if evidence_samples is not None and (not evidence_samples or acquired > evidence_samples[-1].timestamp):
                evidence_samples.append(VerificationFrame(acquired, raw))
            last_sequence = sequence
        if args.view:
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 255 in (27, ord('q'), ord('Q')):
                raise CameraError('User requested emergency stop')
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise CameraError('Viewer closed; emergency stop')
        return raw
    def pause(label, seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            frame(label)
            time.sleep(1/args.hz)
    def safe_status(origin, limits):
        current = client.ptz_status()
        if current['absoluteZoom'] != origin['absoluteZoom']:
            raise CameraError('Zoom changed externally; emergency stop')
        for key in ('azimuth', 'elevation'):
            value = float(current[key])
            lo, hi = limits[key]
            if not lo+40 < value < hi-40:
                raise CameraError('Feedback near reported limit; emergency stop')
            if abs(value-float(origin[key])) > 150:
                raise CameraError('Raw feedback excursion >150; emergency stop')
        return current
    try:
        initial = client.ptz_status()
        report['initial_status'] = initial
        limits = client.ptz_capabilities()['limits_raw']
        safe_status(initial, limits)
        capture = LatestFrameCapture(client.rtsp_source())
        if args.view:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print('Starting real SMOOTH test in 3 seconds. Watch camera/webpage.', flush=True)
        pause('READY - SMOOTH test', 3.)
        for axis, plan in plans.items():
            axis_report = report['axes'][axis]
            before_status = safe_status(initial, limits)
            print('NEXT:', axis.upper(), 'FULL smooth round trip,', round(plan[-1]['t'], 1),
                  'seconds INCLUDING acceleration/deceleration and settling (not per direction)', flush=True)
            pause('NEXT: '+axis.upper()+' smooth round trip', 2.)
            before = frame('BEFORE '+axis.upper())
            cv2.imwrite(str(directory/(axis+'-before.jpg')), before)
            evidence_samples = [VerificationFrame(latest_timestamp, before)]
            ready, armed, done, tripped = (ctx.Event() for _ in range(4))
            heartbeat, results = ctx.Value('d', 0.), ctx.Queue()
            guard = ctx.Process(target=lease_stop_worker, args=(client.host, client._username,
                                client._password, args.http_port, ready, armed, done, tripped,
                                heartbeat, results, plan[-1]['t']+1.))
            guard.start()
            if not ready.wait(6.):
                raise CameraError('Stop watchdog not ready; no motion sent')
            started = previous_tick = time.monotonic()
            with heartbeat.get_lock():
                heartbeat.value = started
            armed.set()
            max_excursion = 0.
            max_frame = before
            max_status = before_status
            feedback = before_status
            next_status = 0.
            count = 0
            max_gap = max_request = 0.
            fields = ['t', 'target', 'speed', 'acceleration', 'jerk', 'pan_command', 'tilt_command',
                      'request_ms', 'azimuth_raw', 'elevation_raw']
            try:
                with (directory/(axis+'-actual.csv')).open('w', newline='', encoding='utf-8') as stream:
                    csv_writer = csv.DictWriter(stream, fieldnames=fields)
                    csv_writer.writeheader()
                    while True:
                        now = time.monotonic()
                        gap = now-previous_tick
                        max_gap = max(max_gap, gap)
                        if gap > .5 or tripped.is_set() or not guard.is_alive():
                            raise CameraError('Control stalled or watchdog tripped; emergency stop')
                        previous_tick = now
                        elapsed = now-started
                        if elapsed >= next_status:
                            feedback = safe_status(initial, limits)
                            next_status = elapsed+.3
                            elapsed = time.monotonic()-started
                        raw = frame(axis.upper()+' SMOOTH  t='+f'{elapsed:.1f}s')
                        key = 'azimuth' if axis == 'pan' else 'elevation'
                        excursion = abs(float(feedback[key])-float(before_status[key]))
                        if excursion > max_excursion:
                            max_excursion, max_frame, max_status = excursion, raw, feedback
                        if tripped.is_set():
                            raise CameraError('Watchdog stopped motion; no restart allowed')
                        row = sample_plan(plan, elapsed)
                        row['t'] = elapsed  # Actual clock time, not clamped trajectory lookup time.
                        command = int(round(row['speed']))
                        pan, tilt = (command, 0) if axis == 'pan' else (0, command)
                        attempted = True  # A lost reply may still have moved the camera.
                        requested = time.monotonic()
                        continuous_move(client, pan, tilt)
                        request_ms = (time.monotonic()-requested)*1000
                        max_request = max(max_request, request_ms)
                        if tripped.is_set():
                            raise CameraError('Watchdog tripped during PUT; emergency stop')
                        with heartbeat.get_lock():
                            heartbeat.value = time.monotonic()
                        csv_writer.writerow(dict(row, pan_command=pan, tilt_command=tilt,
                                                 request_ms=request_ms, azimuth_raw=feedback['azimuth'],
                                                 elevation_raw=feedback['elevation']))
                        stream.flush()
                        count += 1
                        if elapsed >= plan[-1]['t']:
                            break  # Normal completion has already ramped to integer zero.
                        if count % 24 == 0:
                            print(axis.upper(), f't={elapsed:.1f}s command={command} raw={feedback[key]}', flush=True)
                        time.sleep(max(0., 1/args.hz-(time.monotonic()-now)))
            finally:
                axis_report['actual_control_elapsed_seconds'] = time.monotonic()-started
                done.set()
                client.stop()
                report['final_stop_accepted'] = True
                guard.join(4.)
            if guard.is_alive() or tripped.is_set():
                raise CameraError('Stop watchdog failed/tripped; no next axis')
            acknowledgements = []
            while True:
                try:
                    acknowledgements.append(results.get_nowait())
                except queue.Empty:
                    break
            if not any(acknowledgements):
                raise CameraError('Independent stops not acknowledged')
            samples, evidence_samples = evidence_samples, None
            pause(axis.upper()+' STOPPED', .8)
            after_status = safe_status(initial, limits)
            after = frame(axis.upper()+' DONE')
            cv2.imwrite(str(directory/(axis+'-maximum-excursion.jpg')), max_frame)
            cv2.imwrite(str(directory/(axis+'-after.jpg')), after)
            print(axis.upper(), 'STOPPED; verifying adjacent background frames...', flush=True)
            def progress(count, total, verifier):
                print(axis.upper(), f'verification {count}/{total}', flush=True)
                frame(axis.upper()+' STOPPED - verification '+f'{count}/{total}')
            motion, evidence_rows = verify_sequence(samples, axis, detect_foreground, progress)
            with (directory/(axis+'-adjacent-motion.csv')).open('w', newline='', encoding='utf-8') as stream:
                evidence_writer = csv.DictWriter(stream, fieldnames=evidence_rows[0].keys())
                evidence_writer.writeheader()
                evidence_writer.writerows(evidence_rows)
            del samples
            axis_report.update(before=before_status, maximum_excursion_status=max_status, after=after_status,
                               max_raw_excursion=max_excursion, visible_excursion=motion, command_count=count,
                               maximum_tick_gap_seconds=max_gap, maximum_put_ms=max_request,
                               independent_stops=acknowledgements)
            print(json.dumps({axis: axis_report}), flush=True)
            if not motion['reliable']:
                raise CameraError('Adjacent background verification failed: '+motion['reason']+'; no next axis')
        pause('DONE - STOPPED', 2.)
    except CameraError as error:
        report['failure'] = str(error)
    except KeyboardInterrupt:
        report['failure'] = 'User interrupted test'
    except Exception as error:
        report['failure'] = type(error).__name__+' (credential-bearing details omitted)'
    finally:
        if done is not None:
            done.set()
        if attempted:
            for _ in range(3):
                try:
                    client.stop()
                    report['final_stop_accepted'] = True
                except CameraError:
                    report['final_stop_accepted'] = False
                time.sleep(.15)
        if guard is not None:
            guard.join(4.)
        if capture is not None:
            try:
                capture.close()
            except CameraError:
                report['failure'] = report['failure'] or 'Video cleanup failed'
        if video is not None:
            video.release()
        if args.view:
            cv2.destroyAllWindows()
    try:
        report['final_status'] = client.ptz_status()
        time.sleep(.5)
        report['feedback_stationary_after_stop'] = client.ptz_status() == report['final_status']
    except CameraError:
        report['failure'] = report['failure'] or 'Final feedback unavailable'
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('Results:', directory.resolve(), flush=True)
    return int(bool(report['failure']) or not report.get('final_stop_accepted'))


if __name__ == '__main__':
    mp.freeze_support()
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (CameraError, ValueError) as error:
        print('Smooth PTZ test failed:', error)
        raise SystemExit(1)
