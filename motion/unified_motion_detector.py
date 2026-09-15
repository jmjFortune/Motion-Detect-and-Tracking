"""Semantic motion detection for stationary and slowly moving cameras.

YOLO runs once per frame. BoT-SORT and motion classification reuse one GMC.
No PTZ commands are sent by this module.
"""

import sys
import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import ROOT, YAML

_HERE = Path(__file__).resolve().parent
for location in (str(_HERE.parent), str(_HERE)):  # root (hikvision_camera) + motion/ (siblings)
    if location not in sys.path:  # Works both as a package module and as a script.
        sys.path.insert(0, location)

from camera_motion import CameraMotionResult, SharedCameraMotion
from object_motion import (
    MotionConfig, MotionTarget, ObjectMotionClassifier, ObjectObservation,
    SalientTargetSelector, TargetSelection,
)

WINDOW = 'Unified motion detection'
MASK_WINDOW = 'Compensated motion (debug)'
CSV_FIELDS = [
    'timestamp', 'track_id', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2',
    'state', 'reliable', 'speed_px_s', 'normalized_speed', 'residual_dx',
    'residual_dy', 'motion_ratio', 'selected_id', 'selection_status',
    'selected', 'gmc_reliable', 'gmc_reason', 'gmc_points', 'gmc_inlier_ratio',
]


@dataclass
class FrameResult:
    targets: list[MotionTarget]
    selection: TargetSelection
    camera: CameraMotionResult


class UnifiedMotionDetector:
    def __init__(self, model_path='yolo26n.pt', classes='person,dog,cat',
                 imgsz=416, confidence=.35, motion_config=None):
        if imgsz <= 0 or not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError('imgsz must be positive; confidence must be in (0, 1]')
        self.model = YOLO(model_path)
        self.names = self.model.names
        requested = [name.strip() for name in classes.split(',') if name.strip()]
        lookup = {name: index for index, name in self.names.items()}
        unknown = [name for name in requested if name not in lookup]
        if unknown or not requested:
            raise ValueError('Unknown or empty object classes: ' + ', '.join(unknown))
        self.class_ids = sorted({lookup[name] for name in requested})
        self.imgsz = imgsz
        self.confidence = confidence
        self.camera = SharedCameraMotion()
        values = YAML.load(ROOT / 'cfg' / 'trackers' / 'botsort.yaml')
        values.update(with_reid=False, device='cpu')
        self.tracker = BOTSORT(SimpleNamespace(**values))
        self.tracker.gmc = self.camera
        self.motion_config = motion_config or MotionConfig()
        self.classifier = ObjectMotionClassifier(self.motion_config)
        self.selector = SalientTargetSelector()

    def reset(self):
        """Start an independent stream; never call when camera starts moving."""
        self.tracker.reset()
        self.classifier = ObjectMotionClassifier(self.motion_config)
        self.selector = SalientTargetSelector()

    def update_tracks(self, boxes, frame):
        """Return current measured semantic boxes, not predicted/stale boxes.

        Installed Ultralytics track rows are xyxy,id,score,cls,detection-index.
        Using the detection index avoids Kalman smoothing contaminating speed.
        The tracker invokes our shared GMC even when no boxes are present.
        """
        detections = boxes.cpu().numpy()
        # BoT-SORT passes only high-score detections to GMC. Stage the entire
        # allowed semantic set to exclude low-score objects from background too.
        self.camera.set_detections(detections.xyxy)
        rows = self.tracker.update(detections, frame)
        observations = []
        for row in rows:
            if len(row) != 8:
                raise RuntimeError('Unsupported BoT-SORT output: expected 8 columns')
            index = int(row[7])
            if not 0 <= index < len(detections):
                raise RuntimeError('BoT-SORT returned an invalid detection index')
            measured = tuple(float(v) for v in detections.xyxy[index])
            class_id = int(detections.cls[index])
            observations.append(ObjectObservation(
                track_id=int(row[4]), box=measured,
                label=self.names[class_id], confidence=float(detections.conf[index])))
        return observations

    def process(self, frame, timestamp):
        predictions = self.model.predict(
            source=frame, classes=self.class_ids, imgsz=self.imgsz,
            conf=self.confidence, device='cpu', verbose=False)
        observations = self.update_tracks(predictions[0].boxes, frame)
        camera = self.camera.last_result
        targets = self.classifier.update(observations, camera, timestamp)
        selection = self.selector.select(targets, timestamp, frame.shape)
        return FrameResult(targets, selection, camera)


def parse_source(value):
    return int(value) if value.isdecimal() else value


def safe_source_label(source):
    if isinstance(source, int):
        return f'camera index {source}'
    parsed = urlsplit(source)
    if parsed.scheme and '://' in source:
        return 'network video stream'
    return str(source)


def open_capture(source):
    """Keep native OpenCV/FFmpeg messages from printing URL credentials."""
    if isinstance(source, str) and '://' in source:
        # Set before opening the backend. OpenCV's own logs have a separate level.
        level = getattr(cv2, 'getLogLevel', lambda: None)()
        previous = os.environ.get('OPENCV_FFMPEG_LOGLEVEL')
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
        getattr(cv2, 'setLogLevel', lambda value: None)(0)
        try:
            return cv2.VideoCapture(source)
        except cv2.error:
            raise RuntimeError('Cannot open network video stream') from None
        finally:
            if level is not None:
                getattr(cv2, 'setLogLevel', lambda value: None)(level)
            if previous is None:
                os.environ.pop('OPENCV_FFMPEG_LOGLEVEL', None)
            else:
                os.environ['OPENCV_FFMPEG_LOGLEVEL'] = previous
    return cv2.VideoCapture(source)


class FrameClock:
    def __init__(self, is_live, source_fps):
        self.is_live = is_live
        self.source_fps = source_fps
        self.origin = None
        if not is_live and (not math.isfinite(source_fps) or source_fps <= 0):
            raise ValueError('Video FPS is unavailable; use --source-fps for file timing')

    def timestamp(self, frame_index, acquisition_time):
        if not self.is_live:
            return frame_index / self.source_fps
        if self.origin is None:
            self.origin = acquisition_time
        return acquisition_time - self.origin


def resize_to_width(frame, width):
    if width <= 0 or frame.shape[1] <= width:
        return frame
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def annotate(frame, result, fps, show_all=False):
    display = frame.copy()
    selected = result.selection.target
    selected_id = selected.observation.track_id if selected is not None else None
    for target in result.targets:
        moving = target.state == 'MOVING'
        if not moving and not show_all:
            continue
        obs = target.observation
        is_selected = obs.track_id == selected_id
        color = (0, 0, 255) if is_selected else ((0, 220, 255) if moving else (160, 160, 160))
        x1, y1, x2, y2 = (int(round(value)) for value in obs.box)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3 if is_selected else 2)
        quality = '' if target.reliable else ' LOW-QUALITY'
        text = (f'{obs.label} #{obs.track_id} {target.state}{quality} '
                f'{target.speed_px_s:.1f}px/s motion={target.motion_ratio:.2f}')
        cv2.putText(display, text, (max(0, x1), max(16, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    h, w = frame.shape[:2]
    cv2.drawMarker(display, (w//2, h//2), (200, 200, 200), cv2.MARKER_CROSS, 16, 1)
    camera = result.camera
    status = (f'{fps:.1f} FPS | GMC={"OK" if camera.reliable else "UNKNOWN"} '
              f'points={camera.tracked_points} inliers={camera.inlier_ratio:.2f}')
    lock = f'{result.selection.status} ID={result.selection.track_id}'
    for i, text in enumerate((status, f'{lock} | {camera.reason} | Q/Esc: exit')):
        cv2.putText(display, text, (10, 22+i*23), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (70, 255, 70), 1, cv2.LINE_AA)
    return display


def write_csv_frame(writer, timestamp, result):
    camera = result.camera
    common = dict(timestamp=f'{timestamp:.6f}',
                  selected_id=result.selection.track_id if result.selection.track_id is not None else '',
                  selection_status=result.selection.status,
                  gmc_reliable=int(camera.reliable), gmc_reason=camera.reason,
                  gmc_points=camera.tracked_points, gmc_inlier_ratio=camera.inlier_ratio)
    for target in result.targets:
        observation = target.observation
        row = dict(common, track_id=observation.track_id, label=observation.label,
                   confidence=observation.confidence, state=target.state,
                   reliable=int(target.reliable), speed_px_s=target.speed_px_s,
                   normalized_speed=target.normalized_speed, motion_ratio=target.motion_ratio,
                   residual_dx=target.residual[0], residual_dy=target.residual[1],
                   selected=int(result.selection.target is not None and
                                observation.track_id == result.selection.track_id))
        row.update(zip(('x1', 'y1', 'x2', 'y2'), observation.box))
        writer.writerow(row)
    if not result.targets or result.selection.status == 'LOST':
        writer.writerow(dict(common, selected=0))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='Webcam index, video file or RTSP URL')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--classes', default='person,dog,cat')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--resize-width', type=int, default=640, help='Maximum input width')
    parser.add_argument('--source-fps', type=float, help='Override file FPS if metadata is unavailable')
    parser.add_argument('--show-all', action='store_true', help='Also show STATIC/UNKNOWN objects')
    parser.add_argument('--show-mask', action='store_true', help='Show compensated residual mask')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0, help='0: unlimited')
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--start-speed', type=float, default=.08, help='Motion threshold, box diagonals/second')
    parser.add_argument('--stop-speed', type=float, default=.04)
    parser.add_argument('--start-ratio', type=float, default=.06, help='Motion pixel ratio, not probability')
    parser.add_argument('--stop-ratio', type=float, default=.025)
    parser.add_argument('--start-frames', type=int, default=3)
    parser.add_argument('--stop-frames', type=int, default=5)
    return parser


def run(args):
    if args.resize_width <= 0 or args.max_frames < 0:
        raise ValueError('resize-width must be positive; max-frames must be nonnegative')
    if args.source_fps is not None and (not math.isfinite(args.source_fps) or args.source_fps <= 0):
        raise ValueError('source-fps must be positive and finite')
    config = MotionConfig(start_speed=args.start_speed, stop_speed=args.stop_speed,
                          start_ratio=args.start_ratio, stop_ratio=args.stop_ratio,
                          start_frames=args.start_frames, stop_frames=args.stop_frames)
    source = parse_source(args.source)
    capture = None
    csv_file = None
    windows = []
    count = 0
    started = time.perf_counter()
    try:
        capture = open_capture(source)
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video source: {safe_source_label(source)}')
        live = isinstance(source, int) or (isinstance(source, str) and '://' in source)
        fps = args.source_fps if args.source_fps is not None else capture.get(cv2.CAP_PROP_FPS)
        clock = FrameClock(live, fps)
        detector = UnifiedMotionDetector(args.model, args.classes, args.imgsz, args.confidence, config)
        writer = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open('w', newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
        if not args.headless:
            for name in [WINDOW] + ([MASK_WINDOW] if args.show_mask else []):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                windows.append(name)
        started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            acquired = time.monotonic()
            if not ok or frame is None:
                break
            frame = resize_to_width(frame, args.resize_width)
            timestamp = clock.timestamp(count, acquired)
            result = detector.process(frame, timestamp)
            count += 1
            fps = count / max(time.perf_counter()-started, 1e-6)
            if writer is not None:
                write_csv_frame(writer, timestamp, result)
            if not args.headless:
                cv2.imshow(WINDOW, annotate(frame, result, fps, args.show_all))
                if args.show_mask:
                    cv2.imshow(MASK_WINDOW, result.camera.motion_mask)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q'), ord('Q')):
                    break
                try:
                    if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                        break
                except cv2.error:
                    break
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if csv_file is not None:
            csv_file.close()
        for name in windows:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass  # A user may already have closed the window.
    elapsed = max(time.perf_counter()-started, 1e-6)
    print(f'Processed {count} frames, average {count/elapsed:.1f} FPS (this run only).')
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        # Native errors can include source URLs; never echo their text for network input.
        message = 'Network video processing failed; check connection and settings.' if '://' in args.source else str(error)
        print(f'Error: {message}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
