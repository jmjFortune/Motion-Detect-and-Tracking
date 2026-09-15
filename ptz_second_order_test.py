"""Dry-run by default; --execute explicitly moves the physical PTZ camera.

Second-order reference + PD/feedforward, speed/acceleration command limits.
Physical acceleration and jerk are NOT guaranteed without motor calibration.
Timed movement pulses expire at the camera; finally always attempts stop.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

from hikvision_camera import CameraError, add_camera_arguments, client_from_args
from ptz_control import AxisController

FIELDS = ['time', 'axis', 'target', 'reference_position', 'reference_velocity',
          'measured_position', 'measured_velocity', 'command_speed',
          'command_acceleration', 'command']


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--probe', action='store_true', help='Read device status/capabilities, NO movement')
    parser.add_argument('--view', action='store_true', help='Show real camera video during --execute')
    parser.add_argument('--axis', choices=['pan', 'tilt', 'both'], default='pan')
    parser.add_argument('--pan-step', type=float, default=3.)
    parser.add_argument('--tilt-step', type=float, default=2.)
    parser.add_argument('--duration', type=float, default=8.)
    parser.add_argument('--return-to-start', action='store_true')
    parser.add_argument('--hz', type=float, default=10.)
    parser.add_argument('--omega', type=float, default=1.5)
    parser.add_argument('--max-speed', type=float, default=2., help='Nominal degrees/s')
    parser.add_argument('--max-acceleration', type=float, default=1., help='Nominal degrees/s^2')
    parser.add_argument('--pan-gain', type=float, default=3., help='ISAPI units per nominal degree/s; uncalibrated')
    parser.add_argument('--tilt-gain', type=float, default=3.)
    parser.add_argument('--angle-unit', type=float, default=.1, help='Degrees per status unit; verify camera firmware')
    parser.add_argument('--max-excursion', type=float, default=6., help='Abort beyond this observed offset, degrees')
    parser.add_argument('--output-dir', type=Path)
    return parser


def measured_angles(status, unit):
    try:
        positions = {axis: float(status[name])*unit for axis, name in
                     (('pan', 'azimuth'), ('tilt', 'elevation'))}
    except (TypeError, ValueError, KeyError):
        raise CameraError('Camera did not return valid Pan/Tilt positions') from None
    if not all(math.isfinite(v) for v in positions.values()):
        raise CameraError('Nonfinite camera position')
    return positions


def check_args(args):
    for name in ('duration', 'hz', 'omega', 'max_speed', 'max_acceleration',
                 'pan_gain', 'tilt_gain', 'angle_unit', 'max_excursion'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not 5 <= args.hz <= 15 or args.duration > 15 or args.max_speed > 3:
        raise ValueError('Conservative test requires 5..15Hz, <=15 seconds and <=3 degrees/s')
    if args.max_acceleration > 3 or args.max_excursion > 8:
        raise ValueError('Acceleration <=3; observed excursion bound <=8 degrees')
    if not all(math.isfinite(v) and abs(v) <= 4 for v in (args.pan_step, args.tilt_step)):
        raise ValueError('Each test step must be finite and no larger than 4 degrees')
    if max(abs(args.pan_step) if args.axis != 'tilt' else 0.,
           abs(args.tilt_step) if args.axis != 'pan' else 0.) >= args.max_excursion:
        raise ValueError('Step must be strictly inside observed excursion bound')
    if args.view and not args.execute:
        raise ValueError('--view requires --execute (dry-run never connects to video)')
    if args.probe and args.execute:
        raise ValueError('--probe cannot be combined with --execute')


def run(args):
    check_args(args)
    client = None
    capabilities = None
    initial_status = None
    origin = {'pan': 0., 'tilt': 0.}
    if args.execute or args.probe:
        client = client_from_args(args, timeout=.8)
        initial_status = client.ptz_status()
        capabilities = client.ptz_capabilities()
        origin = measured_angles(initial_status, args.angle_unit)
        if args.probe:
            print(json.dumps(dict(device=client.device_info(), status=initial_status,
                                  degrees=origin, capabilities=capabilities, move_requests_sent=0), indent=2))
            return 0
    steps = {'pan': args.pan_step if args.axis != 'tilt' else 0.,
             'tilt': args.tilt_step if args.axis != 'pan' else 0.}
    if args.execute:
        for axis, name in (('pan', 'azimuth'), ('tilt', 'elevation')):
            if not steps[axis]:
                continue
            bounds = capabilities['limits_raw'].get(name)
            if not bounds:
                raise CameraError('Position limits unavailable; refusing physical movement')
            low, high = (v*args.angle_unit for v in bounds)
            if not low+1 <= origin[axis]+steps[axis] <= high-1:
                raise CameraError(f'{axis} target is too close to a reported device limit')
        if args.axis == 'pan' and args.pan_step == 0 or args.axis == 'tilt' and args.tilt_step == 0:
            raise ValueError('Active axis has a zero test step')
    controllers = {axis: AxisController(origin[axis], omega=args.omega,
                      max_speed=args.max_speed, max_acceleration=args.max_acceleration,
                      gain=args.pan_gain if axis == 'pan' else args.tilt_gain)
                   for axis in ('pan', 'tilt')}
    directory = args.output_dir or Path('output') / datetime.now().strftime('ptz-test-%Y%m%d-%H%M%S-%f')
    directory.mkdir(parents=True, exist_ok=False)
    capture = None
    cv2 = None
    window = False
    frame_sequence = 0
    last_image = None
    video_writer = None
    move_attempted = False
    move_requests = 0
    failure = None
    stop_confirmed = None
    positions = origin.copy()
    observed_min = origin.copy()
    observed_max = origin.copy()
    peak_speed = peak_acceleration = 0.
    samples = 0
    dt_nominal = 1./args.hz
    started = last_time = time.monotonic()
    try:
        if args.view:
            import cv2 as camera_cv2
            from ball_camera_detect import LatestFrameCapture
            cv2 = camera_cv2
            capture = LatestFrameCapture(client.rtsp_source())
            cv2.namedWindow('REAL camera - PTZ test (Q: stop)', cv2.WINDOW_NORMAL)
            window = True
            frame_sequence, _, last_image = capture.get()
            cv2.imwrite(str(directory / 'before.jpg'), last_image)
            cv2.imshow('REAL camera - PTZ test (Q: stop)', last_image)
            cv2.waitKey(1)
            height, width = last_image.shape[:2]
            video_writer = cv2.VideoWriter(str(directory / 'physical-test.avi'),
                         cv2.VideoWriter_fourcc(*'MJPG'), args.hz, (width, height))
            if not video_writer.isOpened():
                raise CameraError('Cannot create real-test video recording')
        print('PHYSICAL MOVEMENT ENABLED' if args.execute else 'SIMULATION ONLY: no camera connected')
        print(f'axis={args.axis}; duration={args.duration}s; command cap=6/100; pulse=300ms')
        started = last_time = time.monotonic()
        with (directory / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            iteration = 0
            while True:
                if args.execute:
                    now = time.monotonic()
                    t = now-started
                    dt = dt_nominal if iteration == 0 else now-last_time
                    if t >= args.duration:
                        break
                    if dt > .3:
                        raise CameraError('Control clock exceeded 300ms; aborting rather than sending stale movement')
                    last_time = now
                    positions = measured_angles(client.ptz_status(), args.angle_unit)
                    for axis in ('pan', 'tilt'):
                        offset = positions[axis]-origin[axis]
                        if abs(offset) > args.max_excursion:
                            raise CameraError('Observed displacement exceeded safe test bound')
                        observed_min[axis] = min(observed_min[axis], positions[axis])
                        observed_max[axis] = max(observed_max[axis], positions[axis])
                else:
                    t, dt = iteration*dt_nominal, dt_nominal
                    if t >= args.duration:
                        break
                rows = {}
                for axis in ('pan', 'tilt'):
                    step = 0. if args.return_to_start and t >= args.duration/2. else steps[axis]
                    target = origin[axis]+step
                    rows[axis] = controllers[axis].update(target, positions[axis], dt)
                    if steps[axis] == 0:
                        rows[axis]['command'] = 0
                        rows[axis]['command_speed'] = 0.
                        rows[axis]['command_acceleration'] = 0.
                    writer.writerow(dict(time=t, axis=axis, target=target, **rows[axis]))
                    peak_speed = max(peak_speed, abs(rows[axis]['command_speed']))
                    peak_acceleration = max(peak_acceleration, abs(rows[axis]['command_acceleration']))
                stream.flush()
                if args.execute:
                    if time.monotonic()-now > .25:
                        raise CameraError('Feedback request too slow; refusing a stale pulse')
                    # Mark BEFORE request: even a timed-out reply may have moved.
                    move_attempted = True
                    client.move_pulse(rows['pan']['command'], rows['tilt']['command'])
                    move_requests += 1
                else:
                    for axis in ('pan', 'tilt'):
                        # Ideal velocity plant, not an empirical camera model.
                        positions[axis] += rows[axis]['command_speed']*dt
                        observed_min[axis] = min(observed_min[axis], positions[axis])
                        observed_max[axis] = max(observed_max[axis], positions[axis])
                samples += 1
                if args.view:
                    frame_sequence, _, last_image = capture.get(frame_sequence, timeout=.2)
                    display = last_image.copy()
                    for i, text in enumerate((f'REAL PTZ | pan={positions["pan"]:.1f} tilt={positions["tilt"]:.1f}',
                                             f'command: P={rows["pan"]["command"]} T={rows["tilt"]["command"]} | Q/Esc STOP')):
                        cv2.putText(display, text, (10, 25+i*25), cv2.FONT_HERSHEY_SIMPLEX,
                                    .55, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('REAL camera - PTZ test (Q: stop)', display)
                    video_writer.write(display)
                    if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                        break
                    try:
                        if cv2.getWindowProperty('REAL camera - PTZ test (Q: stop)', cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                if args.execute:
                    # This wait is <=200ms and independent of YOLO inference.
                    remaining = dt_nominal-(time.monotonic()-now)
                    if remaining > 0:
                        time.sleep(remaining)
                iteration += 1
    except Exception as error:
        failure = f'{type(error).__name__}: control/video test aborted (credential-bearing details omitted)'
    except KeyboardInterrupt:
        failure = 'Interrupted by user'
    finally:
        if move_attempted:
            stop_confirmed = False
            for _ in range(2):
                try:
                    client.stop()
                    stop_confirmed = True
                    break
                except CameraError:
                    pass
        if capture is not None:
            try:
                capture.close()
            except CameraError:
                failure = failure or 'Video reader cleanup failed'
        if video_writer is not None:
            video_writer.release()
        if window:
            try:
                cv2.destroyWindow('REAL camera - PTZ test (Q: stop)')
            except cv2.error:
                pass
    final_status = None
    stationary_after_stop = None
    if args.execute:
        try:
            final_status = client.ptz_status()
            positions = measured_angles(final_status, args.angle_unit)
            for axis in ('pan', 'tilt'):
                observed_min[axis] = min(observed_min[axis], positions[axis])
                observed_max[axis] = max(observed_max[axis], positions[axis])
            if move_attempted:
                time.sleep(.4)
                settled_status = client.ptz_status()
                stationary_after_stop = settled_status == final_status
                final_status = settled_status
                positions = measured_angles(final_status, args.angle_unit)
        except CameraError:
            failure = failure or 'Final feedback unavailable'
    if last_image is not None:
        cv2.imwrite(str(directory / 'last.jpg'), last_image)
    report = dict(mode='physical' if args.execute else 'simulation', axis=args.axis,
                  samples=samples, initial_degrees=origin, final_degrees=positions,
                  observed_min=observed_min, observed_max=observed_max,
                  peak_command_speed=peak_speed, peak_command_acceleration=peak_acceleration,
                  move_requests_sent=move_requests, stop_confirmed=stop_confirmed,
                  stationary_after_stop=stationary_after_stop,
                  initial_status_raw=initial_status, final_status_raw=final_status,
                  second_order_omega=args.omega, position_unit_degrees=args.angle_unit,
                  gain_isapi_units_per_degree_s={'pan': args.pan_gain, 'tilt': args.tilt_gain},
                  physical_gain_calibrated=False, physical_jerk_guaranteed=False,
                  failure=failure)
    (directory / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Results: {directory.resolve()}')
    return 1 if failure or (move_attempted and not stop_confirmed) else 0


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except (ValueError, CameraError, OSError) as error:
        print(f'PTZ test failed: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
