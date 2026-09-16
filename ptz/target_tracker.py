"""Predictive image-space target tracking and bounded PTZ velocity control.

The controller operates in normalized image coordinates. Command units are
ISAPI velocity values, not calibrated degrees/second. A Kalman filter predicts
only while fresh measured observations exist; stale/unknown targets cause an
immediate zero command rather than open-loop pursuit.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from ptz.ptz_control import SecondOrderReference, clamp, finite_positive
from hikvision_camera import CameraError, HikvisionClient


@dataclass(frozen=True)
class TrackingObservation:
    timestamp: float
    track_id: int
    box: tuple[float, float, float, float]
    frame_width: int
    frame_height: int
    reliable: bool = True
    state: str = 'MOVING'

    def __post_init__(self):
        values = (*self.box, self.timestamp)
        if (not all(math.isfinite(value) for value in values) or
                type(self.track_id) is not int or self.track_id < 0 or
                type(self.frame_width) is not int or type(self.frame_height) is not int or
                self.frame_width <= 0 or self.frame_height <= 0 or
                self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError('Invalid target observation')

    @property
    def center(self):
        return ((self.box[0]+self.box[2])/2., (self.box[1]+self.box[3])/2.)


class PersistentTargetLock:
    """Keep one tracker ID until it has really disappeared."""

    def __init__(self, lost_grace=2.0):
        finite_positive(lost_grace, 'lost_grace')
        self.lost_grace = lost_grace
        self.track_id = None
        self.last_seen = None
        self.acquired_at = None
        self.last_target = None

    @staticmethod
    def _same_physical_target(previous, candidate):
        if previous.observation.label != candidate.observation.label:
            return False
        a, b = previous.observation.box, candidate.observation.box
        ac = ((a[0]+a[2])/2, (a[1]+a[3])/2)
        bc = ((b[0]+b[2])/2, (b[1]+b[3])/2)
        diagonal = max(math.hypot(a[2]-a[0], a[3]-a[1]), 1.)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        area_ratio = area_b/area_a
        return (math.hypot(bc[0]-ac[0], bc[1]-ac[1]) <= .6*diagonal and
                .4 <= area_ratio <= 2.5)

    def update(self, targets, selected, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Lock timestamp must be finite')
        by_id = {target.observation.track_id: target for target in targets}
        if self.track_id is None:
            if selected is None or not selected.reliable or selected.state != 'MOVING':
                return None, 'WAITING'
            self.track_id = selected.observation.track_id
            self.last_seen = self.acquired_at = timestamp
            self.last_target = selected
            return selected, 'ACQUIRED'
        current = by_id.get(self.track_id)
        if current is not None:
            self.last_seen = timestamp
            self.last_target = current
            return current, 'TRACKING'
        matches = [target for target in targets
                   if self.last_target is not None and
                   self._same_physical_target(self.last_target, target)]
        if matches:
            previous_box = self.last_target.observation.box
            px = (previous_box[0]+previous_box[2])/2
            py = (previous_box[1]+previous_box[3])/2
            current = min(matches, key=lambda target: math.hypot(
                (target.observation.box[0]+target.observation.box[2])/2-px,
                (target.observation.box[1]+target.observation.box[3])/2-py))
            self.track_id = current.observation.track_id
            self.last_seen = timestamp
            self.last_target = current
            return current, 'REASSOCIATED'
        if timestamp-self.last_seen <= self.lost_grace:
            return None, 'HOLDING'
        self.track_id = self.last_seen = self.acquired_at = self.last_target = None
        return None, 'RELEASED'


class ConstantVelocityKalman:
    """2-D [x,y,vx,vy] filter with variable measurement intervals."""

    def __init__(self, process_variance=80., measurement_variance=16.):
        finite_positive(process_variance, 'process_variance')
        finite_positive(measurement_variance, 'measurement_variance')
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.reset()

    def reset(self):
        self.state = None
        self.covariance = None
        self.timestamp = None

    def _predict_in_place(self, timestamp):
        dt = timestamp-self.timestamp
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Kalman timestamps must increase')
        f = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        g = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]], dtype=float)
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + self.process_variance*(g @ g.T)
        self.timestamp = timestamp

    def update(self, measurement, timestamp):
        point = np.asarray(measurement, dtype=float).reshape(-1)
        if point.size != 2 or not np.isfinite(point).all() or not math.isfinite(timestamp):
            raise ValueError('Invalid Kalman measurement')
        if self.state is None:
            self.state = np.array([point[0], point[1], 0., 0.])
            self.covariance = np.diag([self.measurement_variance]*2+[400., 400.])
            self.timestamp = timestamp
            return tuple(self.state)
        self._predict_in_place(timestamp)
        h = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
        innovation = point-h @ self.state
        s = h @ self.covariance @ h.T + np.eye(2)*self.measurement_variance
        gain = self.covariance @ h.T @ np.linalg.inv(s)
        self.state += gain @ innovation
        identity = np.eye(4)
        # Joseph form keeps covariance symmetric and positive under rounding.
        kh = identity-gain @ h
        self.covariance = kh @ self.covariance @ kh.T + gain @ (np.eye(2)*self.measurement_variance) @ gain.T
        return tuple(self.state)

    def forecast(self, horizon):
        if self.state is None or not math.isfinite(horizon) or horizon < 0:
            raise ValueError('Kalman filter is uninitialized or horizon is invalid')
        return (float(self.state[0]+self.state[2]*horizon),
                float(self.state[1]+self.state[3]*horizon))

    @property
    def velocity(self):
        if self.state is None:
            raise ValueError('Kalman filter is uninitialized')
        return float(self.state[2]), float(self.state[3])


@dataclass
class JerkLimitedAxis:
    max_speed: float = 24.
    max_acceleration: float = 18.
    max_jerk: float = 45.
    max_deceleration: float | None = None
    max_braking_jerk: float | None = None
    response: float = 5.
    speed: float = 0.
    acceleration: float = 0.

    def __post_init__(self):
        if self.max_deceleration is None:
            self.max_deceleration = self.max_acceleration
        if self.max_braking_jerk is None:
            self.max_braking_jerk = self.max_jerk
        for value, name in ((self.max_speed, 'max_speed'),
                            (self.max_acceleration, 'max_acceleration'),
                            (self.max_jerk, 'max_jerk'),
                            (self.max_deceleration, 'max_deceleration'),
                            (self.max_braking_jerk, 'max_braking_jerk'),
                            (self.response, 'response')):
            finite_positive(value, name)

    def update(self, desired, dt):
        finite_positive(dt, 'dt')
        if not math.isfinite(desired):
            raise ValueError('Desired command must be finite')
        desired = clamp(desired, self.max_speed)
        braking = (abs(self.speed) > .01 and
                   (desired*self.speed <= 0 or abs(desired) < abs(self.speed)))
        acceleration_limit = self.max_deceleration if braking else self.max_acceleration
        jerk_limit = self.max_braking_jerk if braking else self.max_jerk
        target_acceleration = clamp(
            self.response**2*(desired-self.speed)-2*self.response*self.acceleration,
            acceleration_limit)
        previous_acceleration = self.acceleration
        self.acceleration += clamp(target_acceleration-self.acceleration, jerk_limit*dt)
        self.acceleration = clamp(self.acceleration,
                                  max(self.max_acceleration, self.max_deceleration))
        self.speed = clamp(self.speed+self.acceleration*dt, self.max_speed)
        return self.speed, self.acceleration, (self.acceleration-previous_acceleration)/dt

    def reset(self):
        self.speed = self.acceleration = 0.


@dataclass
class TrackingConfig:
    deadband_x: float = .08
    deadband_y: float = .10
    prediction_horizon: float = .10
    observation_timeout: float = .50
    error_omega: float = 7.
    pan_limit: int = 50
    tilt_limit: int = 50
    minimum_command: float = 8.
    motor_floor: int = 15
    motor_start_threshold: float = 2.
    motor_stop_threshold: float = .75
    position_gain: float = 1.25
    velocity_gain: float = .45
    max_acceleration: float = 40.
    max_jerk: float = 100.
    max_deceleration: float = 110.
    max_braking_jerk: float = 300.
    static_hold_x: float = .10
    static_hold_y: float = .12
    static_release_x: float = .16
    static_release_y: float = .18

    def __post_init__(self):
        values = (self.deadband_x, self.deadband_y, self.prediction_horizon,
                  self.observation_timeout, self.error_omega, self.minimum_command,
                  self.max_acceleration, self.max_jerk,
                  self.motor_start_threshold, self.motor_stop_threshold,
                  self.position_gain, self.velocity_gain,
                  self.max_deceleration, self.max_braking_jerk,
                  self.static_hold_x, self.static_hold_y,
                  self.static_release_x, self.static_release_y)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('Tracking limits must be positive and finite')
        if self.deadband_x >= 1 or self.deadband_y >= 1:
            raise ValueError('Deadbands must be below one normalized half-frame')
        if (not self.static_hold_x < self.static_release_x < 1 or
                not self.static_hold_y < self.static_release_y < 1):
            raise ValueError('Static hold thresholds require valid hysteresis')
        if (type(self.pan_limit) is not int or type(self.tilt_limit) is not int or
                not 1 <= self.pan_limit <= 60 or not 1 <= self.tilt_limit <= 60 or
                self.minimum_command > min(self.pan_limit, self.tilt_limit) or
                type(self.motor_floor) is not int or
                not 1 <= self.motor_floor <= min(self.pan_limit, self.tilt_limit) or
                self.motor_stop_threshold >= self.motor_start_threshold):
            raise ValueError('Unsafe PTZ command limits')


@dataclass(frozen=True)
class TrackingDecision:
    pan: int
    tilt: int
    measured_center: tuple[float, float] | None
    predicted_center: tuple[float, float] | None
    error_x: float
    error_y: float
    radial_error: float
    track_id: int | None
    emergency_stop: bool
    reason: str
    normalized_target_velocity: tuple[float, float] | None = None


class PredictiveTrackingController:
    def __init__(self, config=None):
        self.config = config or TrackingConfig()
        self.kalman = ConstantVelocityKalman()
        self.error_x = SecondOrderReference(omega=self.config.error_omega)
        self.error_y = SecondOrderReference(omega=self.config.error_omega)
        self.pan = JerkLimitedAxis(
            max_speed=self.config.pan_limit,
            max_acceleration=self.config.max_acceleration,
            max_jerk=self.config.max_jerk,
            max_deceleration=self.config.max_deceleration,
            max_braking_jerk=self.config.max_braking_jerk)
        self.tilt = JerkLimitedAxis(
            max_speed=self.config.tilt_limit,
            max_acceleration=self.config.max_acceleration,
            max_jerk=self.config.max_jerk,
            max_deceleration=self.config.max_deceleration,
            max_braking_jerk=self.config.max_braking_jerk)
        self.track_id = None
        self.last_measurement_time = None
        self._pan_active = self._tilt_active = False
        self._pan_static_hold = self._tilt_static_hold = False

    def reset(self):
        self.kalman.reset()
        self.error_x = SecondOrderReference(omega=self.config.error_omega)
        self.error_y = SecondOrderReference(omega=self.config.error_omega)
        self.pan.reset()
        self.tilt.reset()
        self.track_id = self.last_measurement_time = None
        self._pan_active = self._tilt_active = False
        self._pan_static_hold = self._tilt_static_hold = False

    def _static_hold(self, error, enter, release, axis):
        name = f'_{axis}_static_hold'
        holding = getattr(self, name)
        if holding:
            holding = abs(error) < release
        elif abs(error) <= enter:
            holding = True
        setattr(self, name, holding)
        if holding:
            actuator = self.pan if axis == 'pan' else self.tilt
            actuator.reset()
            setattr(self, f'_{axis}_active', False)
        return holding

    def _actuate(self, logical, limit, axis):
        active_name = f'_{axis}_active'
        active = getattr(self, active_name)
        magnitude = abs(logical)
        if active and magnitude <= self.config.motor_stop_threshold:
            active = False
        elif not active and magnitude >= self.config.motor_start_threshold:
            active = True
        setattr(self, active_name, active)
        if not active:
            return 0
        span = max(limit-self.config.motor_start_threshold, 1e-9)
        fraction = min(1., max(0., magnitude-self.config.motor_start_threshold)/span)
        command = self.config.motor_floor+(limit-self.config.motor_floor)*fraction
        return int(math.copysign(round(command), logical))

    def _desired(self, error, error_rate, deadband, limit):
        magnitude = abs(error)
        # A moving target at the center still receives velocity feed-forward.
        if magnitude <= deadband and abs(error_rate) <= .03:
            return 0.
        position_term = 0. if magnitude <= deadband else math.copysign(
            (magnitude-deadband)/(1.-deadband), error)
        normalized = (self.config.position_gain*position_term+
                      self.config.velocity_gain*clamp(error_rate, 2.))
        if abs(normalized) < .02:
            return 0.
        value = self.config.minimum_command+(limit-self.config.minimum_command)*min(1., abs(normalized))
        return math.copysign(value, normalized)

    def _stop(self, reason):
        self.reset()
        return TrackingDecision(0, 0, None, None, 0., 0., 0., None, True, reason)

    def update(self, observation, now, dt):
        finite_positive(dt, 'dt')
        if not math.isfinite(now):
            raise ValueError('Control time must be finite')
        if observation is None:
            return self._stop('no target')
        age = now-observation.timestamp
        if age < 0 or age > self.config.observation_timeout:
            return self._stop('stale target')
        if not observation.reliable or observation.state not in ('MOVING', 'TRACKING', 'LOCKED_STATIC'):
            return self._stop('unreliable or non-moving target')
        if self.track_id != observation.track_id:
            self.reset()
            self.track_id = observation.track_id
        if self.last_measurement_time != observation.timestamp:
            if self.last_measurement_time is not None and observation.timestamp <= self.last_measurement_time:
                return self._stop('out-of-order target')
            self.kalman.update(observation.center, observation.timestamp)
            self.last_measurement_time = observation.timestamp
        locked_static = observation.state == 'LOCKED_STATIC'
        predicted = (observation.center if locked_static else
                     self.kalman.forecast(age+self.config.prediction_horizon))
        velocity_x, velocity_y = ((0., 0.) if locked_static else self.kalman.velocity)
        ex = (predicted[0]-observation.frame_width/2)/(observation.frame_width/2)
        ey = (predicted[1]-observation.frame_height/2)/(observation.frame_height/2)
        velocity_x /= observation.frame_width/2
        velocity_y /= observation.frame_height/2
        filtered_x, _ = self.error_x.update(clamp(ex, 1.5), dt)
        filtered_y, _ = self.error_y.update(clamp(ey, 1.5), dt)
        if not locked_static:
            self._pan_static_hold = self._tilt_static_hold = False
        hold_pan = locked_static and self._static_hold(
            ex, self.config.static_hold_x, self.config.static_release_x, 'pan')
        hold_tilt = locked_static and self._static_hold(
            ey, self.config.static_hold_y, self.config.static_release_y, 'tilt')
        desired_pan = (0. if hold_pan else self._desired(
            filtered_x, velocity_x, self.config.deadband_x, self.config.pan_limit))
        # Device mapping verified on this camera: positive tilt moves view upward.
        desired_tilt = (0. if hold_tilt else -self._desired(
            filtered_y, velocity_y, self.config.deadband_y, self.config.tilt_limit))
        pan, _, _ = self.pan.update(desired_pan, dt)
        tilt, _, _ = self.tilt.update(desired_tilt, dt)
        physical_pan = self._actuate(pan, self.config.pan_limit, 'pan')
        physical_tilt = self._actuate(tilt, self.config.tilt_limit, 'tilt')
        return TrackingDecision(physical_pan, physical_tilt, observation.center,
                                predicted, ex, ey, math.hypot(ex, ey), observation.track_id,
                                False, ('static center hold' if hold_pan and hold_tilt else
                                        'second-order PD + velocity feed-forward'),
                                (velocity_x, velocity_y))


class ObservationMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._observation = None

    def publish(self, observation):
        with self._lock:
            self._observation = observation

    def snapshot(self):
        with self._lock:
            return self._observation


class PositionGuard:
    def __init__(self, initial, limits, max_excursion=150., margin=40.):
        self.initial = {key: float(initial[key]) for key in ('azimuth', 'elevation')}
        self.limits = limits
        if max_excursion is not None:
            finite_positive(max_excursion, 'max_excursion')
        finite_positive(margin, 'margin')
        self.max_excursion, self.margin = max_excursion, margin

    def apply(self, pan, tilt, status):
        azimuth, elevation = float(status['azimuth']), float(status['elevation'])
        amin, amax = self.limits['azimuth']
        emin, emax = self.limits['elevation']
        blocked = []
        # Measured mapping: positive pan decreases azimuth; positive tilt increases elevation.
        if ((azimuth <= amin+self.margin and pan > 0) or
                (azimuth >= amax-self.margin and pan < 0) or
                (self.max_excursion is not None and
                 azimuth-self.initial['azimuth'] <= -self.max_excursion and pan > 0) or
                (self.max_excursion is not None and
                 azimuth-self.initial['azimuth'] >= self.max_excursion and pan < 0)):
            pan, blocked = 0, blocked+['pan']
        if ((elevation <= emin+self.margin and tilt < 0) or
                (elevation >= emax-self.margin and tilt > 0) or
                (self.max_excursion is not None and
                 elevation-self.initial['elevation'] <= -self.max_excursion and tilt < 0) or
                (self.max_excursion is not None and
                 elevation-self.initial['elevation'] >= self.max_excursion and tilt > 0)):
            tilt, blocked = 0, blocked+['tilt']
        return pan, tilt, blocked


def watchdog_stop_worker(host, user, password, port, ready, armed, done, tripped,
                         heartbeat, results, hard_duration, lease=.8):
    """Independent process repeatedly stops PTZ if the controller stalls."""
    try:
        client = HikvisionClient(host, user, password, port, timeout=.5)
        client.ptz_status()
        ready.set()
        if not armed.wait(10.):
            return
        start = time.monotonic()
        while not done.is_set():
            with heartbeat.get_lock():
                last = heartbeat.value
            now = time.monotonic()
            if now-last > lease or (hard_duration is not None and now-start > hard_duration):
                tripped.set()
                break
            time.sleep(.03)
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
