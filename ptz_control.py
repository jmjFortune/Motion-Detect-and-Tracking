"""Second-order reference model and bounded velocity-command tracking.

This is a software controller, not a claim about camera motor dynamics.
Degrees/s -> ISAPI speed units must be calibrated on the actual camera.
"""

import math
from dataclasses import dataclass


def finite_positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


def clamp(value, bound):
    return max(-bound, min(bound, value))


@dataclass
class SecondOrderReference:
    """Exact critically damped x''+2*w*x'+w*w*(x-target)=0.

    A changing target never resets position or velocity. The exact update is
    stable at variable dt; it does not assume a fixed detector frame rate.
    """

    position: float = 0.
    velocity: float = 0.
    omega: float = 1.5

    def __post_init__(self):
        finite_positive(self.omega, 'omega')
        if not all(math.isfinite(v) for v in (self.position, self.velocity)):
            raise ValueError('Initial state must be finite')

    def update(self, target, dt):
        finite_positive(dt, 'dt')
        if not math.isfinite(target):
            raise ValueError('target must be finite')
        error = self.position-target
        coefficient = self.velocity+self.omega*error
        decay = math.exp(-self.omega*dt)
        self.position = target+(error+coefficient*dt)*decay
        self.velocity = (self.velocity-self.omega*coefficient*dt)*decay
        return self.position, self.velocity


@dataclass
class VelocityRamp:
    max_speed: float = 2.
    max_acceleration: float = 1.
    speed: float = 0.

    def __post_init__(self):
        finite_positive(self.max_speed, 'max_speed')
        finite_positive(self.max_acceleration, 'max_acceleration')
        if not math.isfinite(self.speed) or abs(self.speed) > self.max_speed:
            raise ValueError('Invalid initial speed')

    def update(self, desired, dt):
        finite_positive(dt, 'dt')
        if not math.isfinite(desired):
            raise ValueError('desired speed must be finite')
        previous = self.speed
        desired = clamp(desired, self.max_speed)
        self.speed += clamp(desired-self.speed, self.max_acceleration*dt)
        return self.speed, (self.speed-previous)/dt


class AxisController:
    def __init__(self, position=0., omega=1.5, kp=.8, kd=.15,
                 max_speed=2., max_acceleration=1., gain=3., max_command=6):
        for value, name in ((gain, 'gain'), (max_command, 'max_command')):
            finite_positive(value, name)
        if max_command > 10 or int(max_command) != max_command:
            raise ValueError('Test command cap must be an integer <=10')
        if not all(math.isfinite(v) and v >= 0 for v in (kp, kd)):
            raise ValueError('PD gains must be finite and nonnegative')
        self.reference = SecondOrderReference(position, omega=omega)
        # Respect normalized command cap before rounding, not only afterward.
        self.ramp = VelocityRamp(min(max_speed, max_command/gain), max_acceleration)
        self.kp, self.kd, self.gain = kp, kd, gain
        self.max_command = int(max_command)
        self.previous_measured = position
        self.measured_velocity = 0.

    def update(self, target, measured, dt):
        finite_positive(dt, 'dt')
        if not math.isfinite(measured):
            raise ValueError('Measured position must be finite')
        position, velocity = self.reference.update(target, dt)
        raw_velocity = (measured-self.previous_measured)/dt
        alpha = 1.-math.exp(-dt/.2)
        self.measured_velocity += alpha*(raw_velocity-self.measured_velocity)
        self.previous_measured = measured
        desired = velocity+self.kp*(position-measured)+self.kd*(velocity-self.measured_velocity)
        command_speed, acceleration = self.ramp.update(desired, dt)
        command = int(round(clamp(command_speed*self.gain, self.max_command)))
        return dict(reference_position=position, reference_velocity=velocity,
                    measured_position=measured, measured_velocity=self.measured_velocity,
                    command_speed=command_speed, command_acceleration=acceleration,
                    command=command)
