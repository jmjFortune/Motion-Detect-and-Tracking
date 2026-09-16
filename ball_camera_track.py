"""Predictive moving-target PTZ tracking for the real Hikvision camera.

Without --execute this creates only a controller simulation and never connects
to a camera. Real mode is bounded in time and excursion, uses an independent
stop watchdog, and never pursues stale or predicted-only targets.
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ball_camera_detect import LatestFrameCapture
from hikvision_camera import CameraError, HikvisionClient, add_camera_arguments, client_from_args
from motion.unified_motion_detector import (
    CSV_FIELDS, UnifiedMotionDetector, annotate, resize_to_width, write_csv_frame,
)
from motion.object_motion import TargetSelection
from ptz.target_tracker import (
    ObservationMailbox, PersistentTargetLock, PositionGuard, PredictiveTrackingController,
    TrackingConfig, TrackingObservation, watchdog_stop_worker,
)


WINDOW = 'REAL moving-target tracking - Q/Esc: STOP'
CONTROL_FIELDS = [
    'time', 'track_id', 'pan', 'tilt', 'error_x', 'error_y', 'radial_error',
    'measured_x', 'measured_y', 'predicted_x', 'predicted_y', 'reason',
    'emergency_stop', 'blocked_axes', 'azimuth_raw', 'elevation_raw',
]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--execute', action='store_true', help='Actually connect and move PTZ')
    parser.add_argument('--continuous', action='store_true',
                        help='Run until Q/Esc, window close, or Ctrl+C')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--channel', type=int, default=102)
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--classes', default='person,dog,cat')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--resize-width', type=int, default=640)
    parser.add_argument('--duration', type=float, default=30., help='Maximum real run time, seconds')
    parser.add_argument('--tracking-seconds', type=float, default=10.)
    parser.add_argument('--control-hz', type=float, default=10.)
    parser.add_argument('--deadband-x', type=float, default=.05)
    parser.add_argument('--deadband-y', type=float, default=.07)
    parser.add_argument('--prediction-horizon', type=float, default=.10)
    parser.add_argument('--pan-limit', type=int, default=50)
    parser.add_argument('--tilt-limit', type=int, default=50)
    parser.add_argument('--max-excursion', type=float, default=150.)
    parser.add_argument('--sharpness-threshold', type=float, default=80.)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show-all', action='store_true')
    return parser


def check_args(args):
    numbers = (args.duration, args.tracking_seconds, args.control_hz,
               args.max_excursion, args.sharpness_threshold)
    if not all(math.isfinite(value) and value > 0 for value in numbers):
        raise ValueError('Durations, rates and limits must be positive and finite')
    if not 5 <= args.duration <= 60 or not 1 <= args.tracking_seconds <= args.duration:
        raise ValueError('Duration must be 5..60s and tracking-seconds within duration')
    if not 5 <= args.control_hz <= 15 or args.resize_width < 1:
        raise ValueError('Control rate must be 5..15Hz and width positive')
    if not 20 <= args.max_excursion <= 300:
        raise ValueError('Initial-position excursion bound must be 20..300 raw units')


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_error_curve(rows, path):
    canvas = np.full((650, 1200, 3), 250, np.uint8)
    left, right, top, bottom = 85, 1160, 55, 575
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
    cv2.putText(canvas, 'Tracking error curve (pixels)', (left, 32),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    cv2.putText(canvas, 'time (s)', (right-80, bottom+48),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
    if not rows:
        cv2.putText(canvas, 'No valid target observations', (390, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 180), 2)
        return cv2.imwrite(str(path), canvas)
    times = np.array([row['time'] for row in rows], float)
    ex = np.array([row['error_x_px'] for row in rows], float)
    ey = np.array([row['error_y_px'] for row in rows], float)
    radial = np.hypot(ex, ey)
    span = max(float(times[-1]-times[0]), 1e-6)
    bound = max(10., float(np.max(np.abs(np.concatenate((ex, ey, radial)))))*1.1)
    zero_y = round((top+bottom)/2)
    cv2.line(canvas, (left, zero_y), (right, zero_y), (170, 170, 170), 1)
    for value in (-bound, 0., bound):
        y = round(zero_y-value/bound*(bottom-top)/2)
        cv2.putText(canvas, f'{value:.0f}', (12, y+5), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (50, 50, 50), 1)
    def points(values):
        return np.array([[round(left+(t-times[0])/span*(right-left)),
                          round(zero_y-v/bound*(bottom-top)/2)]
                         for t, v in zip(times, values)], np.int32)
    for values, color, label, x in ((ex, (30, 30, 220), 'X', 90),
                                     (ey, (220, 70, 30), 'Y', 145),
                                     (radial, (20, 150, 150), 'radial', 200)):
        cv2.polylines(canvas, [points(values)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, 620), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    return cv2.imwrite(str(path), canvas)


class PTZControlThread(threading.Thread):
    def __init__(self, client, mailbox, controller, guard, done, tripped,
                 heartbeat, hz=10.):
        super().__init__(name='ptz-control', daemon=True)
        self.client, self.mailbox, self.controller = client, mailbox, controller
        self.guard, self.done, self.tripped, self.heartbeat = guard, done, tripped, heartbeat
        self.period = 1./hz
        self.rows = []
        self.latest = None
        self.failure = None
        self.stop_accepted = False
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return self.latest

    def run(self):
        previous = time.monotonic()
        next_status = previous
        status = {'azimuth': str(self.guard.initial['azimuth']),
                  'elevation': str(self.guard.initial['elevation'])}
        last_sent = None
        last_request = 0.
        origin = previous
        try:
            while not self.done.is_set():
                started = now = time.monotonic()
                if self.tripped.is_set():
                    raise CameraError('Independent stop watchdog tripped')
                dt = min(.2, max(.01, now-previous))
                previous = now
                decision = self.controller.update(self.mailbox.snapshot(), now, dt)
                if now >= next_status:
                    status = self.client.ptz_status()
                    next_status = time.monotonic()+.5
                pan, tilt, blocked = self.guard.apply(decision.pan, decision.tilt, status)
                if blocked:
                    # Per-axis outward motion is refused; return direction remains available.
                    decision_reason = decision.reason+'; boundary blocked '+','.join(blocked)
                else:
                    decision_reason = decision.reason
                command = (pan, tilt)
                if command != last_sent or now-last_request >= .25 or decision.emergency_stop:
                    if command == (0, 0):
                        self.client.stop()
                    else:
                        self.client.continuous_move(pan, tilt,
                                                    command_limit=max(self.controller.config.pan_limit,
                                                                      self.controller.config.tilt_limit))
                    last_sent, last_request = command, time.monotonic()
                with self.heartbeat.get_lock():
                    self.heartbeat.value = time.monotonic()
                row = dict(time=now-origin, track_id=decision.track_id or '', pan=pan, tilt=tilt,
                           error_x=decision.error_x, error_y=decision.error_y,
                           radial_error=decision.radial_error,
                           measured_x='' if decision.measured_center is None else decision.measured_center[0],
                           measured_y='' if decision.measured_center is None else decision.measured_center[1],
                           predicted_x='' if decision.predicted_center is None else decision.predicted_center[0],
                           predicted_y='' if decision.predicted_center is None else decision.predicted_center[1],
                           reason=decision_reason, emergency_stop=int(decision.emergency_stop),
                           blocked_axes=';'.join(blocked), azimuth_raw=status['azimuth'],
                           elevation_raw=status['elevation'])
                self.rows.append(row)
                with self._lock:
                    self.latest = decision
                time.sleep(max(0., self.period-(time.monotonic()-started)))
        except Exception as error:
            self.failure = f'{type(error).__name__}: PTZ control stopped (details omitted)'
            self.done.set()
        finally:
            try:
                self.client.stop()
                self.stop_accepted = True
            except CameraError:
                self.stop_accepted = False


def simulation(args, directory):
    controller = PredictiveTrackingController(TrackingConfig(
        args.deadband_x, args.deadband_y, args.prediction_horizon,
        pan_limit=args.pan_limit, tilt_limit=args.tilt_limit))
    errors, controls = [], []
    for index in range(150):
        timestamp = index*.1
        x = 500-180*min(1., timestamp/8.)
        y = 330-90*min(1., timestamp/8.)
        obs = TrackingObservation(timestamp, 1, (x-40, y-80, x+40, y+80), 640, 480)
        decision = controller.update(obs, timestamp, .1)
        controls.append(dict(time=timestamp, pan=decision.pan, tilt=decision.tilt,
                             error_x=decision.error_x, error_y=decision.error_y))
        errors.append(dict(time=timestamp, error_x_px=x-320, error_y_px=y-240))
    draw_error_curve(errors, directory/'tracking-error.png')
    report = dict(mode='simulation', camera_connections=0, ptz_move_requests_sent=0,
                  kalman_prediction=True, stable_tracking_required_seconds=args.tracking_seconds,
                  clear_capture_created=False, control_samples=len(controls))
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DRY RUN: simulation only; no camera connection or PTZ request.')
    print('Results:', directory.resolve())
    return 0


def run(args):
    check_args(args)
    directory = args.output_dir or Path('output')/datetime.now().strftime('ball-track-%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    if not args.execute:
        return simulation(args, directory)
    config = TrackingConfig(deadband_x=args.deadband_x, deadband_y=args.deadband_y,
                            prediction_horizon=args.prediction_horizon,
                            pan_limit=args.pan_limit, tilt_limit=args.tilt_limit)
    # Load/warm inference before arming any PTZ output.
    detector = UnifiedMotionDetector(args.model, args.classes, args.imgsz, args.confidence)
    client = client_from_args(args, timeout=.6)
    device = client.device_info()
    initial = client.ptz_status()
    capabilities = client.ptz_capabilities()['limits_raw']
    control_client = HikvisionClient(client.host, client._username, client._password,
                                     args.http_port, timeout=.5)
    capture = LatestFrameCapture(client.rtsp_source(args.channel))
    mailbox, controller = ObservationMailbox(), PredictiveTrackingController(config)
    target_lock = PersistentTargetLock(lost_grace=2.)
    # Continuous tracking may traverse the full mechanical range. Finite runs
    # retain the conservative initial-position excursion bound.
    guard = PositionGuard(initial, capabilities,
                          None if args.continuous else args.max_excursion)
    ctx = mp.get_context('spawn')
    ready, armed, watchdog_done, tripped = (ctx.Event() for _ in range(4))
    heartbeat, stop_results = ctx.Value('d', 0.), ctx.Queue()
    hard_duration = None if args.continuous else args.duration+3.
    watchdog = ctx.Process(target=watchdog_stop_worker,
        args=(client.host, client._username, client._password, args.http_port,
              ready, armed, watchdog_done, tripped, heartbeat, stop_results, hard_duration))
    watchdog.start()
    if not ready.wait(6.):
        capture.close()
        raise CameraError('Stop watchdog did not become ready; no movement started')
    done = threading.Event()
    with heartbeat.get_lock():
        heartbeat.value = time.monotonic()
    armed.set()
    worker = PTZControlThread(control_client, mailbox, controller, guard, done,
                              tripped, heartbeat, args.control_hz)
    worker.start()
    created_window = False
    sequence = frames = skipped = 0
    started = time.monotonic()
    origin = None
    stable_id = None
    active_tracking = 0.
    last_valid_acquired = None
    max_stable = 0.
    stable_tracking_passed = False
    centered_streak = 0
    best_capture = None
    best_sharpness = -1.
    capture_pass = False
    error_rows = []
    moving_ids = set()
    states = Counter()
    lock_states = Counter()
    visual_motion_frames = 0
    max_visual_shift_px = 0.
    last_curve_write = 0.
    failure = None
    last_display = None
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            created_window = True
        with (directory/'detections.csv').open('w', newline='', encoding='utf-8-sig') as detection_stream:
            detection_writer = csv.DictWriter(detection_stream, fieldnames=CSV_FIELDS)
            detection_writer.writeheader()
            while not done.is_set():
                new_sequence, acquired, raw = capture.get(sequence)
                skipped += max(0, new_sequence-sequence-1)
                sequence = new_sequence
                if origin is None:
                    origin = acquired
                timestamp = acquired-origin
                frame = resize_to_width(raw, args.resize_width)
                result = detector.process(frame, timestamp)
                frames += 1
                for target in result.targets:
                    states[target.state] += 1
                    if target.state == 'MOVING':
                        moving_ids.add(target.observation.track_id)
                selected, lock_status = target_lock.update(
                    result.targets, result.selection.target, timestamp)
                lock_states[lock_status] += 1
                result.selection = TargetSelection(target_lock.track_id, selected, lock_status)
                write_csv_frame(detection_writer, timestamp, result)
                detection_stream.flush()
                # Motion classification selects the initial target. Once locked,
                # the current detector box remains controllable through STATIC or
                # temporarily unreliable GMC frames.
                valid = selected is not None
                if valid:
                    control_state = ('LOCKED_STATIC' if selected.state == 'STATIC'
                                     else 'TRACKING')
                    observation = TrackingObservation(
                        acquired, selected.observation.track_id, selected.observation.box,
                        frame.shape[1], frame.shape[0], True, control_state)
                    mailbox.publish(observation)
                else:
                    mailbox.publish(None)
                visible_id = None if selected is None else selected.observation.track_id
                if valid:
                    if visible_id != stable_id:
                        stable_id, active_tracking = visible_id, 0.
                    elif last_valid_acquired is not None:
                        active_tracking += min(.5, acquired-last_valid_acquired)
                    last_valid_acquired = acquired
                    max_stable = max(max_stable, active_tracking)
                    if active_tracking >= args.tracking_seconds:
                        stable_tracking_passed = True
                else:
                    last_valid_acquired = None
                    active_tracking = 0.
                    if lock_status == 'RELEASED':
                        stable_id = None
                if selected is not None:
                    cx, cy = ((selected.observation.box[0]+selected.observation.box[2])/2,
                              (selected.observation.box[1]+selected.observation.box[3])/2)
                    ex, ey = cx-frame.shape[1]/2, cy-frame.shape[0]/2
                    error_rows.append(dict(time=timestamp, track_id=selected.observation.track_id,
                                           error_x_px=ex, error_y_px=ey,
                                           radial_error_px=math.hypot(ex, ey)))
                    decision = worker.snapshot()
                    low_command = decision is None or max(abs(decision.pan), abs(decision.tilt)) <= 4
                    centered = (abs(ex) <= args.deadband_x*frame.shape[1]/2 and
                                abs(ey) <= args.deadband_y*frame.shape[0]/2 and low_command and
                                selected.observation.label == 'person')
                    centered_streak = centered_streak+1 if centered else 0
                    if centered_streak >= 3 and not capture_pass:
                        score = sharpness(raw)
                        best_sharpness = max(best_sharpness, score)
                        if score >= args.sharpness_threshold:
                            best_capture = raw.copy()
                            capture_pass = bool(cv2.imwrite(
                                str(directory/'center-capture.jpg'), best_capture))
                else:
                    centered_streak = 0
                last_display = annotate(frame, result, frames/max(time.monotonic()-started, 1e-6),
                                        args.show_all)
                decision = worker.snapshot()
                if decision is not None:
                    if (result.camera.reliable and
                            (decision.pan != 0 or decision.tilt != 0)):
                        center = (frame.shape[1]/2, frame.shape[0]/2)
                        warped_center = result.camera.transform_point(center)
                        visual_shift = math.hypot(warped_center[0]-center[0],
                                                  warped_center[1]-center[1])
                        max_visual_shift_px = max(max_visual_shift_px, visual_shift)
                        if visual_shift >= .75:
                            visual_motion_frames += 1
                    text = (f'PTZ pan={decision.pan:+d} tilt={decision.tilt:+d} '
                            f'err=({decision.error_x:+.2f},{decision.error_y:+.2f}) '
                            f'v={decision.normalized_target_velocity} '
                            f'pred={decision.predicted_center}')
                    cv2.putText(last_display, text, (10, last_display.shape[0]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
                    if decision.predicted_center is not None:
                        point = tuple(int(round(v)) for v in decision.predicted_center)
                        image_center = (last_display.shape[1]//2, last_display.shape[0]//2)
                        cv2.line(last_display, image_center, point, (255, 0, 255), 2,
                                 cv2.LINE_AA)
                        cv2.drawMarker(last_display, point, (255, 0, 255),
                                      cv2.MARKER_DIAMOND, 14, 2)
                progress_text = (f'STABLE {min(active_tracking, args.tracking_seconds):.1f}/'
                                 f'{args.tracking_seconds:.1f}s ' +
                                 ('PASS' if stable_tracking_passed else ''))
                cv2.putText(last_display, progress_text, (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 200, 0) if stable_tracking_passed else (0, 220, 255),
                            2, cv2.LINE_AA)
                cv2.putText(last_display,
                            'CAPTURE SAVED' if capture_pass else 'CAPTURE WAITING',
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, .65,
                            (0, 200, 0) if capture_pass else (0, 220, 255),
                            2, cv2.LINE_AA)
                if selected is not None:
                    x1, y1, x2, y2 = map(int, selected.observation.box)
                    cv2.rectangle(last_display, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(last_display, f'LOCK ID {target_lock.track_id} {lock_status}',
                                (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                .55, (255, 0, 255), 2, cv2.LINE_AA)
                if not args.headless:
                    cv2.imshow(WINDOW, last_display)
                    if cv2.waitKey(1) & 255 in (27, ord('q'), ord('Q')):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if not args.continuous and stable_tracking_passed and capture_pass:
                    break
                if timestamp-last_curve_write >= 2. and error_rows:
                    draw_error_curve(error_rows, directory/'tracking-error.png')
                    last_curve_write = timestamp
                if not args.continuous and time.monotonic()-started >= args.duration:
                    break
    except (CameraError, RuntimeError, cv2.error) as error:
        failure = f'{type(error).__name__}: tracking stopped (details omitted)'
    except KeyboardInterrupt:
        failure = 'Interrupted by user'
    finally:
        mailbox.publish(None)
        done.set()
        worker.join(3.)
        watchdog_done.set()
        watchdog.join(4.)
        for _ in range(3):
            try:
                client.stop()
            except CameraError:
                pass
            time.sleep(.15)
        try:
            capture.close()
        except CameraError:
            failure = failure or 'Video reader cleanup failed'
        if created_window:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
    if worker.is_alive():
        failure = failure or 'PTZ control thread did not stop'
    if watchdog.is_alive() or tripped.is_set():
        failure = failure or 'Independent PTZ watchdog failed or tripped'
    stop_acks = []
    while True:
        try:
            stop_acks.append(stop_results.get_nowait())
        except queue.Empty:
            break
    final = settled = None
    try:
        final = client.ptz_status()
        time.sleep(.5)
        settled = client.ptz_status()
    except CameraError:
        failure = failure or 'Final PTZ feedback unavailable'
    capture_pass = bool(capture_pass and (directory/'center-capture.jpg').is_file())
    if last_display is not None:
        cv2.imwrite(str(directory/'last.jpg'), last_display)
    draw_error_curve(error_rows, directory/'tracking-error.png')
    if error_rows:
        with (directory/'tracking-error.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=error_rows[0].keys())
            writer.writeheader()
            writer.writerows(error_rows)
    with (directory/'control.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=CONTROL_FIELDS)
        writer.writeheader()
        writer.writerows(worker.rows)
    commanded_motion = any(row['pan'] != 0 or row['tilt'] != 0 for row in worker.rows)
    visual_motion_verified = visual_motion_frames >= 3 if commanded_motion else None
    report = dict(mode='physical', device=device, frames=frames,
                  elapsed_seconds=time.monotonic()-started, skipped_stream_frames=skipped,
                  moving_ids=sorted(moving_ids), states=dict(states),
                  target_lock_states=dict(lock_states), locked_target_id=stable_id,
                  stable_tracking_required_seconds=args.tracking_seconds,
                  longest_same_target_seconds=max_stable,
                  stable_tracking_achieved=stable_tracking_passed,
                  kalman_prediction=True, prediction_horizon_seconds=args.prediction_horizon,
                  controller='second-order PD plus Kalman velocity feed-forward',
                  error_samples=len(error_rows), error_curve='tracking-error.png',
                  clear_capture_created=capture_pass,
                  clear_capture_file='center-capture.jpg' if capture_pass else None,
                  capture_sharpness=best_sharpness if best_capture is not None else None,
                  capture_threshold=args.sharpness_threshold,
                  capture_threshold_pass=capture_pass,
                  initial_status=initial, final_status=settled,
                  feedback_stationary_after_stop=final == settled if final is not None else None,
                  commanded_motion=commanded_motion,
                  visually_verified_camera_motion=visual_motion_verified,
                  visual_motion_evidence_frames=visual_motion_frames,
                  maximum_background_warp_pixels=max_visual_shift_px,
                  control_samples=len(worker.rows), stop_accepted=worker.stop_accepted,
                  position_guard=('mechanical-limits-only' if args.continuous
                                  else f'initial-position-plus-minus-{args.max_excursion}'),
                  independent_stop_acknowledgements=stop_acks,
                  physical_speed_calibrated=False, physical_jerk_guaranteed=False,
                  failure=failure or worker.failure)
    (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Results:', directory.resolve())
    return int(bool(report['failure']) or not report['stable_tracking_achieved'] or
               not report['clear_capture_created'] or
               (commanded_motion and not visual_motion_verified))


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except (ValueError, CameraError, OSError, cv2.error) as error:
        print('Target tracking failed:', error)
        return 1


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
