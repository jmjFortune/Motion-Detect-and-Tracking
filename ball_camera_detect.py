"""Real PTZ-camera detection test. This program NEVER sends ISAPI PUT/move.

Reads fresh RTSP frames on a separate thread to avoid inference queue latency.
The password comes from PTZ_PASSWORD/CAMERA_PASSWORD (process environment or the
local .env file) or a hidden prompt, never from a source argument.
"""

import argparse
import csv
import json
import os
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from hikvision_camera import CameraError, add_camera_arguments, client_from_args
from motion.unified_motion_detector import (
    CSV_FIELDS, UnifiedMotionDetector, annotate, resize_to_width, write_csv_frame,
)

WINDOW = 'Ball camera - detection only (NO PTZ)'


class LatestFrameCapture:
    def __init__(self, source):
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._sequence = 0
        self._timestamp = 0.0
        self._alive = True
        self._level = getattr(cv2, 'getLogLevel', lambda: None)()
        self._options = {key: os.environ.get(key) for key in
                         ('OPENCV_FFMPEG_LOGLEVEL', 'OPENCV_FFMPEG_CAPTURE_OPTIONS')}
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        getattr(cv2, 'setLogLevel', lambda level: None)(0)
        try:
            self._capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000])
            if not self._capture.isOpened():
                self._capture.release()
                raise CameraError('Cannot open camera RTSP channel; check credentials and stream configuration')
        except Exception:
            self._restore_logs()
            raise
        self._thread = threading.Thread(target=self._read, name='camera-latest-frame', daemon=True)
        self._thread.start()

    def _restore_logs(self):
        if self._level is not None:
            getattr(cv2, 'setLogLevel', lambda level: None)(self._level)
        for key, value in self._options.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _read(self):
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                timestamp = time.monotonic()
                if not ok or frame is None:
                    break
                with self._condition:
                    self._frame = frame
                    self._timestamp = timestamp
                    self._sequence += 1
                    self._condition.notify_all()
        finally:
            self._capture.release()
            with self._condition:
                self._alive = False
                self._condition.notify_all()

    def get(self, after=0, timeout=5.0):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._sequence > after or not self._alive, timeout=timeout)
            if not ready or self._sequence <= after:
                raise CameraError('Camera stream stopped or no fresh frame arrived')
            return self._sequence, self._timestamp, self._frame.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=4.0)
        if self._thread.is_alive():
            # Never race native read() with release(), which can crash OpenCV.
            raise CameraError('Camera reader did not stop within its configured timeout')
        self._restore_logs()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    parser.add_argument('--channel', type=int, default=102)
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--classes', default='person,dog,cat')
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--resize-width', type=int, default=640)
    parser.add_argument('--duration', type=float, default=0., help='Seconds; 0 means until Q/Esc')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--show-all', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    return parser


def run(args):
    if args.duration < 0 or args.max_frames < 0 or args.resize_width < 1:
        raise ValueError('Invalid duration, max-frames or resize-width')
    client = client_from_args(args, timeout=4.)
    device = client.device_info()
    before = client.ptz_status()
    detector = UnifiedMotionDetector(args.model, args.classes, args.imgsz, args.confidence)
    directory = args.output_dir or Path('output') / datetime.now().strftime('ball-detect-%Y%m%d-%H%M%S')
    directory.mkdir(parents=True, exist_ok=False)
    capture = None
    created_window = False
    states = Counter()
    labels = Counter()
    tracked_ids = set()
    moving_ids = set()
    count = reliable_frames = 0
    best_score = -1.
    last_display = best_display = None
    started = time.monotonic()
    sequence = 0
    skipped = 0
    failure = None
    try:
        capture = LatestFrameCapture(client.rtsp_source(args.channel))
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            created_window = True
        with (directory / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            started = time.monotonic()
            origin = None
            while True:
                new_sequence, acquired, frame = capture.get(sequence)
                skipped += max(0, new_sequence-sequence-1)
                sequence = new_sequence
                if origin is None:
                    origin = acquired
                frame = resize_to_width(frame, args.resize_width)
                result = detector.process(frame, acquired-origin)
                count += 1
                reliable_frames += int(result.camera.reliable)
                write_csv_frame(writer, acquired-origin, result)
                stream.flush()
                for target in result.targets:
                    states[target.state] += 1
                    labels[target.observation.label] += 1
                    tracked_ids.add(target.observation.track_id)
                    if target.state == 'MOVING':
                        moving_ids.add(target.observation.track_id)
                elapsed = max(time.monotonic()-started, 1e-6)
                last_display = annotate(frame, result, count/elapsed, args.show_all)
                score = sum(target.state == 'MOVING' for target in result.targets)
                score += .01 * len(result.targets)
                if score > best_score:
                    best_score = score
                    best_display = last_display.copy()
                if not args.headless:
                    cv2.imshow(WINDOW, last_display)
                    if cv2.waitKey(1) & 0xff in (27, ord('q'), ord('Q')):
                        break
                    try:
                        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break
                if args.max_frames and count >= args.max_frames:
                    break
                if args.duration and elapsed >= args.duration:
                    break
    except (CameraError, RuntimeError, cv2.error):
        failure = 'Camera/video processing failed (credential-bearing details omitted)'
    except KeyboardInterrupt:
        failure = 'Interrupted by user'
    finally:
        if capture is not None:
            capture.close()
        if created_window:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
    elapsed = max(time.monotonic()-started, 1e-6)
    try:
        after = client.ptz_status()
    except CameraError:
        after = None
    for name, image in (('best.jpg', best_display), ('last.jpg', last_display)):
        if image is not None and not cv2.imwrite(str(directory / name), image):
            raise OSError('Cannot write annotated detection image')
    report = dict(device=device, channel=args.channel, frames=count,
                  elapsed_seconds=elapsed, processed_fps=count/elapsed,
                  skipped_stream_frames=skipped, gmc_reliable_frames=reliable_frames,
                  tracked_ids=sorted(tracked_ids), moving_ids=sorted(moving_ids),
                  states=dict(states), labels=dict(labels),
                  ptz_before=before, ptz_after=after,
                  reported_ptz_unchanged=before == after if after is not None else None,
                  ptz_move_requests_sent=0, failure=failure)
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Results: {directory.resolve()}')
    return 1 if failure or not count else 0


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except (ValueError, CameraError, OSError, cv2.error):
        print('Camera test failed; check IP, password, output directory and stream settings.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
