"""Camera-compensated visual motion and an observation-only salient ID lock.

Speeds are image pixels/s and current-box diagonals/s, not physical speeds.
Pair evidence requires continuous measured observations and majority valid
overlap (>= half the full rasterized box, >= 16 pixels). New-view pixels
never provide stillness evidence. No tracker predictions or control I/O.
"""

from dataclasses import dataclass
import math

import numpy as np

from camera_motion import CameraMotionResult


def _nonnegative(value, name):
    try:
        valid = math.isfinite(value) and value >= 0
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite and nonnegative')


@dataclass
class ObjectObservation:
    track_id: int
    box: tuple[float, float, float, float]
    label: str
    confidence: float


@dataclass
class MotionConfig:
    start_speed: float = 0.08
    stop_speed: float = 0.04
    start_ratio: float = 0.06
    stop_ratio: float = 0.025
    start_frames: int = 3
    stop_frames: int = 5
    unreliable_grace: float = 0.5
    history_ttl: float = 2.0

    def __post_init__(self):
        for name in ('start_speed', 'stop_speed', 'start_ratio', 'stop_ratio',
                     'unreliable_grace', 'history_ttl'):
            _nonnegative(getattr(self, name), name)
        if self.stop_speed >= self.start_speed or self.stop_ratio >= self.start_ratio:
            raise ValueError('stop thresholds must be strictly below start thresholds')
        for name in ('start_frames', 'stop_frames'):
            value = getattr(self, name)
            _nonnegative(value, name)
            if isinstance(value, (bool, np.bool_)) or value < 1 or int(value) != value:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class MotionTarget:
    observation: ObjectObservation
    state: str
    reliable: bool
    speed_px_s: float
    normalized_speed: float
    motion_ratio: float
    residual: tuple[float, float]
    moving_duration: float


@dataclass
class TargetSelection:
    track_id: int | None
    target: MotionTarget | None
    status: str


@dataclass
class _History:
    last_seen: float
    previous_center: tuple[float, float] | None = None
    previous_time: float | None = None
    previous_reliable: bool = False
    last_reliable: float | None = None
    state: str = 'UNKNOWN'
    start_count: int = 0
    stop_count: int = 0
    moving_duration: float = 0.0

    def break_evidence(self):
        self.start_count = self.stop_count = 0


def _geometry(box):
    values = np.asarray(box, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = map(float, values)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1+x2)/2, (y1+y2)/2), math.hypot(x2-x1, y2-y1)


def _supported(box, camera):
    valid = camera.valid_mask
    if valid.ndim != 2 or camera.motion_mask.shape != valid.shape:
        return False
    height, width = valid.shape
    x1, y1, x2, y2 = box
    left, top, right, bottom = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    full_area = (right-left) * (bottom-top)
    region = valid[max(0, min(height, top)):max(0, min(height, bottom)),
                   max(0, min(width, left)):max(0, min(width, right))]
    count = np.count_nonzero(region)
    return count >= 16 and count >= full_area * 0.5


class ObjectMotionClassifier:
    def __init__(self, config=None):
        self.config = MotionConfig() if config is None else config
        self.histories: dict[int, _History] = {}

    def _expire_quality(self, history, timestamp):
        if (history.last_reliable is None or
                timestamp-history.last_reliable > self.config.unreliable_grace):
            history.state = 'UNKNOWN'
            history.moving_duration = 0.0

    def update(self, observations, camera: CameraMotionResult,
               timestamp: float) -> list[MotionTarget]:
        if not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite')
        observations = list(observations)
        visible = {observation.track_id for observation in observations}
        for track_id, history in list(self.histories.items()):
            if timestamp-history.last_seen > self.config.history_ttl:
                del self.histories[track_id]
            elif track_id not in visible:
                history.previous_center = history.previous_time = None
                history.previous_reliable = False
                history.break_evidence()
                self._expire_quality(history, timestamp)

        targets = []
        for observation in observations:
            history = self.histories.setdefault(observation.track_id, _History(timestamp))
            if not history.previous_reliable:
                self._expire_quality(history, timestamp)
            geometry = _geometry(observation.box)
            dt = 0.0 if history.previous_time is None else timestamp-history.previous_time
            reliable = bool(geometry is not None and camera.reliable and dt > 0 and
                            history.previous_center is not None and
                            _supported(observation.box, camera))
            residual, speed, normalized, ratio = (0.0, 0.0), 0.0, 0.0, 0.0
            if reliable:
                predicted = camera.transform_point(history.previous_center)
                center, diagonal = geometry
                residual = (center[0]-predicted[0], center[1]-predicted[1])
                speed = math.hypot(*residual) / dt
                normalized = speed / diagonal
                ratio = camera.motion_ratio(observation.box)
                reliable = all(math.isfinite(v) for v in (*residual, speed, normalized, ratio))
            if reliable:
                old_state = history.state
                history.last_reliable = timestamp
                if normalized >= self.config.start_speed or ratio >= self.config.start_ratio:
                    history.start_count += 1
                    history.stop_count = 0
                    if history.start_count >= self.config.start_frames:
                        history.state = 'MOVING'
                elif normalized <= self.config.stop_speed and ratio <= self.config.stop_ratio:
                    history.stop_count += 1
                    history.start_count = 0
                    if history.stop_count >= self.config.stop_frames:
                        history.state = 'STATIC'
                else:
                    history.break_evidence()
                if history.state != 'MOVING':
                    history.moving_duration = 0.0
                elif old_state == 'MOVING' and history.previous_reliable:
                    history.moving_duration += dt
            else:
                residual, speed, normalized, ratio = (0.0, 0.0), 0.0, 0.0, 0.0
                history.break_evidence()
                self._expire_quality(history, timestamp)

            # Camera warps describe the immediately preceding image, including
            # failed pairs. Keep raw current centers, never compensated ones.
            history.previous_center = None if geometry is None else geometry[0]
            history.previous_time = timestamp
            history.previous_reliable = reliable
            history.last_seen = timestamp
            targets.append(MotionTarget(observation, history.state, reliable, speed,
                                        normalized, ratio, residual, history.moving_duration))
        return targets


class SalientTargetSelector:
    def __init__(self, lost_grace=0.7):
        _nonnegative(lost_grace, 'lost_grace')
        self.lost_grace = lost_grace
        self._track_id = None
        self._last_good = None

    @staticmethod
    def _score(target, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = target.observation.box
        area = max(0., min(width, x2)-max(0., x1)) * max(0., min(height, y2)-max(0., y1))
        speed = max(0., target.normalized_speed)
        duration = max(0., target.moving_duration)
        return (min(1., area/(height*width)) + speed/(1.+speed) +
                min(1., max(0., target.motion_ratio)) + duration/(1.+duration)) / 4.

    def select(self, targets, timestamp, frame_shape) -> TargetSelection:
        if not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite')
        targets = list(targets)
        current = next((target for target in targets
                        if target.observation.track_id == self._track_id), None)
        if self._track_id is not None:
            if current is not None and current.state == 'MOVING' and current.reliable:
                self._last_good = timestamp
                return TargetSelection(self._track_id, current, 'TRACKING')
            in_grace = timestamp-self._last_good <= self.lost_grace
            if in_grace and current is None:
                return TargetSelection(self._track_id, None, 'LOST')
            if (in_grace and current is not None and current.state == 'MOVING'
                    and not current.reliable):
                return TargetSelection(self._track_id, current, 'TRACKING')
            self._track_id = self._last_good = None

        candidates = [target for target in targets if target.reliable and target.state == 'MOVING']
        if not candidates:
            return TargetSelection(None, None, 'NONE')
        target = max(candidates, key=lambda item: (self._score(item, frame_shape),
                                                 -item.observation.track_id))
        self._track_id = target.observation.track_id
        self._last_good = timestamp
        return TargetSelection(self._track_id, target, 'TRACKING')
