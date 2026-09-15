"""Background-only, adjacent-frame evidence for real PTZ test excursions.

Failed pairs break the composition chain: never bridge a visibility/time gap,
never substitute PTZ feedback for video, and never relax GMC quality gates.
Inference/verification runs AFTER the camera stops, outside its control loop.
"""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera_motion import SharedCameraMotion


@dataclass
class VerificationFrame:
    timestamp: float
    frame: np.ndarray


class BackgroundMotionVerifier:
    def __init__(self, axis='pan', motion=None, max_gap=.4):
        if axis not in ('pan', 'tilt') or not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError('Invalid verification axis or time gap')
        self.axis, self.max_gap = axis, max_gap
        self.motion = motion if motion is not None else SharedCameraMotion()
        self.previous_time = None
        self.shape = None
        self.rows = []
        self.reasons = Counter()
        self.runs = []
        self._break_chain()

    def _break_chain(self):
        self.composed = np.eye(3, dtype=np.float64)
        self.run_pairs = 0
        self.run_duration = 0.
        self.run_peak = 0.

    @staticmethod
    def exclusion_boxes(frame, boxes):
        h, w = frame.shape[:2]
        boxes = np.asarray(boxes, dtype=float)
        if not boxes.size:
            boxes = np.empty((0, 4), dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError('Foreground boxes must be finite Nx4 xyxy coordinates')
        padded = boxes + np.array([-4., -4., 4., 4.])
        # Time/status text is attached to the image, not to the physical scene.
        # Scale margins with image height to support smaller regression fixtures.
        top = max(1, round(h*.105))
        bottom = max(1, round(h*.09))
        overlay = np.array([[0, 0, w, top], [0, h-bottom, w, h]], dtype=float)
        return np.vstack((padded, overlay))

    def observe(self, frame, boxes, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Verification timestamp must be finite')
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError('Verification requires strictly increasing fresh timestamps')
        dt = 0. if self.previous_time is None else timestamp-self.previous_time
        changed = self.shape is not None and frame.shape[:2] != self.shape
        gap = self.previous_time is not None and dt > self.max_gap
        if changed or gap:
            self.motion.reset_params()
            self._break_chain()
        excluded = self.exclusion_boxes(frame, boxes)
        self.motion.apply(frame, excluded)
        result = self.motion.last_result
        reason = ('timestamp gap' if gap else 'frame shape changed') if changed or gap else result.reason
        reliable = bool(result.reliable and not changed and not gap)
        if reliable:
            warp = np.eye(3, dtype=np.float64)
            warp[:2] = result.warp
            if not np.isfinite(warp).all():
                reliable, reason = False, 'nonfinite warp'
        if reliable:
            self.composed = warp @ self.composed  # previous -> current, not addition of translations
            h, w = frame.shape[:2]
            center = np.array([w/2., h/2., 1.])
            displacement = (self.composed @ center)[:2]-center[:2]
            self.run_pairs += 1
            self.run_duration += dt
            axis_displacement = displacement[0 if self.axis == 'pan' else 1]
            self.run_peak = max(self.run_peak, abs(float(axis_displacement)))
            if self.run_pairs == 1:
                self.runs.append({})
            self.runs[-1].update(pairs=self.run_pairs, duration=self.run_duration,
                                 max_axis_excursion_pixels=self.run_peak)
        else:
            self._break_chain()
            displacement = np.array([0., 0.])
            self.reasons[reason] += 1
        row = dict(timestamp=timestamp, reliable=reliable, reason=reason,
                   tracked_points=result.tracked_points, inlier_ratio=result.inlier_ratio,
                   coverage=result.coverage, foreground_boxes=len(boxes),
                   chain_dx=float(displacement[0]), chain_dy=float(displacement[1]),
                   chain_pairs=self.run_pairs, chain_duration=self.run_duration)
        self.rows.append(row)
        self.previous_time, self.shape = timestamp, frame.shape[:2]
        return row

    def summary(self):
        count = len(self.rows)
        good = sum(row['reliable'] for row in self.rows)
        fraction = good/max(1, count-1)
        supported = [run for run in self.runs if run['pairs'] >= 5 and run['duration'] >= .4]
        peak = max((run['max_axis_excursion_pixels'] for run in supported), default=0.)
        verified = fraction >= .5 and peak >= 3.
        reason = 'verified adjacent background motion' if verified else (
            'insufficient reliable background pairs' if fraction < .5 else
            'insufficient continuous background excursion')
        return dict(reliable=verified, method='adjacent background LK/RANSAC composition', axis=self.axis,
                    frames=count, reliable_pairs=good, reliable_fraction=fraction,
                    max_axis_excursion_pixels=peak, reason=reason,
                    unreliable_reasons=dict(self.reasons), continuous_runs=self.runs,
                    requires_foreground_exclusion=True, feedback_used_as_video_proof=False)


def create_foreground_detector(model_path='yolo26n.pt'):
    # Avoid hidden downloads or model loads while a camera is already moving.
    if not Path(model_path).is_file():
        raise ValueError('Local verification model is missing: '+str(model_path))
    from ultralytics import YOLO
    model = YOLO(model_path)
    classes = [index for index, name in model.names.items() if name in ('person', 'dog', 'cat')]
    if not any(name == 'person' for name in model.names.values()):
        raise ValueError('Verification model must contain the person class')
    def detect(frame):
        result = model.predict(frame, imgsz=416, conf=.25, classes=classes,
                               device='cpu', verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy()
    detect(np.zeros((240, 320, 3), dtype=np.uint8))
    return detect


def verify_sequence(samples, axis, detect_foreground, progress=None):
    verifier = BackgroundMotionVerifier(axis)
    for index, sample in enumerate(samples):
        boxes = detect_foreground(sample.frame)
        verifier.observe(sample.frame, boxes, sample.timestamp)
        if progress is not None and (index % 20 == 0 or index+1 == len(samples)):
            progress(index+1, len(samples), verifier)
    return verifier.summary(), verifier.rows
