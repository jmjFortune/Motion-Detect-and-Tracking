"""S-curve target-speed generation and critically damped second-order smoothing.

Units are ISAPI command units, NOT calibrated physical degrees/second.
The S curve bounds target speed and its first two time derivatives. A positive
critically damped filter smooths that target further; dense numerical checks
verify the resulting float command. Integer/device motion is not jerk-certified.
"""

import sys
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:  # ptz/ modules import root-level and motion/ modules.
    sys.path.insert(0, PROJECT_ROOT)

from ptz.ptz_control import SecondOrderReference, finite_positive


@dataclass
class SpeedTransition:
    start: float
    end: float
    max_acceleration: float = 30.
    max_jerk: float = 60.

    def __post_init__(self):
        finite_positive(self.max_acceleration, 'max_acceleration')
        finite_positive(self.max_jerk, 'max_jerk')
        if not all(math.isfinite(v) for v in (self.start, self.end)):
            raise ValueError('Transition speeds must be finite')
        delta = abs(self.end-self.start)
        self.sign = 1. if self.end >= self.start else -1.
        self.tj = min(math.sqrt(delta/self.max_jerk), self.max_acceleration/self.max_jerk)
        self.ta = 0. if not delta else max(0., delta/(self.max_jerk*self.tj)-self.tj)
        self.duration = 2*self.tj+self.ta

    def sample(self, t):
        if not math.isfinite(t):
            raise ValueError('Time must be finite')
        if t <= 0 or self.duration == 0:
            return self.start, 0., 0.
        if t >= self.duration:
            return self.end, 0., 0.
        j, tj, ta = self.max_jerk*self.sign, self.tj, self.ta
        if t < tj:
            return self.start+.5*j*t*t, j*t, j
        v1, a1 = self.start+.5*j*tj*tj, j*tj
        if t < tj+ta:
            return v1+a1*(t-tj), a1, 0.
        dt = t-tj-ta
        return v1+a1*ta+a1*dt-.5*j*dt*dt, a1-j*dt, -j


class SmoothRoundTrip:
    def __init__(self, peak=24., omega=4., max_acceleration=30., max_jerk=60.,
                 first_sign=-1, cruise_seconds=.3):
        finite_positive(peak, 'peak')
        finite_positive(omega, 'omega')
        if not math.isfinite(cruise_seconds) or not .3 <= cruise_seconds <= 3.:
            raise ValueError('Cruise hold must be .3..3 seconds')
        if peak > 30 or first_sign not in (-1, 1):
            raise ValueError('Peak must be <=30; first_sign must be +/-1')
        self.peak, self.omega = peak, omega
        self.max_acceleration, self.max_jerk = max_acceleration, max_jerk
        self.segments = []
        time_cursor, speed = 0., 0.
        # Two smooth rest-to-rest moves, opposite directions. No hard stop between updates.
        for target, hold in ((first_sign*peak, cruise_seconds), (0., 1.),
                             (-first_sign*peak, cruise_seconds), (0., 2.)):
            transition = SpeedTransition(speed, target, max_acceleration, max_jerk)
            self.segments.append((time_cursor, time_cursor+transition.duration, transition))
            time_cursor += transition.duration
            self.segments.append((time_cursor, time_cursor+hold, target))
            time_cursor += hold
            speed = target
        self.duration = time_cursor

    def target(self, t):
        for start, end, segment in self.segments:
            if start <= t < end:
                return segment.sample(t-start)[0] if isinstance(segment, SpeedTransition) else segment
        return 0.

    def generate(self, dt=.005):
        finite_positive(dt, 'dt')
        if dt > .02:
            raise ValueError('Numerical integration step must be <=20ms')
        state = SecondOrderReference(omega=self.omega)
        # Here position is smoothed target SPEED, and velocity its time derivative.
        def derivative(t, x, a):
            return a, self.omega**2*(self.target(t)-x)-2*self.omega*a
        rows = []
        t = 0.
        while True:
            x, a = state.position, state.velocity
            jerk = derivative(t, x, a)[1]
            rows.append(dict(t=t, target=self.target(t), speed=x, acceleration=a, jerk=jerk))
            if t >= self.duration:
                break
            h = min(dt, self.duration-t)
            k1 = derivative(t, x, a)
            k2 = derivative(t+h/2, x+h*k1[0]/2, a+h*k1[1]/2)
            k3 = derivative(t+h/2, x+h*k2[0]/2, a+h*k2[1]/2)
            k4 = derivative(t+h, x+h*k3[0], a+h*k3[1])
            state.position += h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
            state.velocity += h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
            t = min(self.duration, t+h)
        validate_plan(rows, self.peak, self.max_acceleration, self.max_jerk)
        return rows


def validate_plan(rows, peak, max_acceleration, max_jerk):
    for row in rows:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError('Nonfinite trajectory')
        for key, bound in [('speed', peak), ('acceleration', max_acceleration), ('jerk', max_jerk)]:
            if abs(row[key]) > bound+1e-4:
                raise ValueError('Trajectory exceeds '+key+' limit')
    if round(rows[0]['speed']) != 0 or round(rows[-1]['speed']) != 0:
        raise ValueError('Trajectory must begin and finish with zero integer command')


def sample_plan(rows, t):
    import bisect
    index = bisect.bisect_right(rows, t, key=lambda row: row['t'])-1
    if index < 0:
        return rows[0].copy()
    if index >= len(rows)-1:
        return rows[-1].copy()
    a, b = rows[index], rows[index+1]
    fraction = (t-a['t'])/(b['t']-a['t'])
    return {key: a[key]+fraction*(b[key]-a[key]) for key in a}
