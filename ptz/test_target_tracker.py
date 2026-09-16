import contextlib
import math
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hikvision_camera import CameraError, HikvisionClient
from ptz.target_tracker import (
    ConstantVelocityKalman, JerkLimitedAxis, PersistentTargetLock, PositionGuard,
    PredictiveTrackingController, TrackingConfig, TrackingObservation,
    watchdog_stop_worker,
)
from motion.object_motion import MotionTarget, ObjectObservation


def observation(x=320., y=240., timestamp=1., track_id=1, reliable=True, state='MOVING'):
    return TrackingObservation(timestamp, track_id, (x-40, y-80, x+40, y+80),
                               640, 480, reliable, state)


def motion_target(track_id, state='MOVING', reliable=True):
    return MotionTarget(ObjectObservation(track_id, (10, 10, 50, 90), 'person', .9),
                        state, reliable, 10., .2, .2, (2., 1.), 1.)


class PredictiveControllerTests(unittest.TestCase):
    def test_kalman_predicts_constant_velocity_ahead(self):
        kalman = ConstantVelocityKalman(process_variance=1., measurement_variance=.01)
        for i in range(8):
            kalman.update((100+i*5, 80-i*2), i*.1)
        x, y = kalman.forecast(.2)
        self.assertAlmostEqual(x, 145., delta=2.)
        self.assertAlmostEqual(y, 62., delta=2.)

    def test_center_deadband_converges_to_zero(self):
        controller = PredictiveTrackingController()
        for i in range(80):
            decision = controller.update(observation(timestamp=1+i*.1), 1+i*.1, .1)
        self.assertEqual((decision.pan, decision.tilt), (0, 0))
        self.assertFalse(decision.emergency_stop)

    def test_direction_mapping_and_simultaneous_axes(self):
        controller = PredictiveTrackingController()
        for i in range(25):
            decision = controller.update(observation(540, 400, 1+i*.1), 1+i*.1, .1)
        self.assertGreater(decision.pan, 0)
        self.assertLess(decision.tilt, 0)
        controller = PredictiveTrackingController()
        for i in range(25):
            decision = controller.update(observation(100, 80, 1+i*.1), 1+i*.1, .1)
        self.assertLess(decision.pan, 0)
        self.assertGreater(decision.tilt, 0)

    def test_nonzero_motor_commands_clear_measured_dead_zone(self):
        controller = PredictiveTrackingController()
        commands = []
        for i in range(25):
            decision = controller.update(observation(540, 400, 1+i*.1), 1+i*.1, .1)
            commands.extend((decision.pan, decision.tilt))
        self.assertTrue(any(commands))
        self.assertTrue(all(abs(value) >= 15 for value in commands if value))

    def test_kalman_velocity_drives_target_before_large_position_error(self):
        with_feedforward = PredictiveTrackingController(
            TrackingConfig(velocity_gain=.5, position_gain=.1))
        without_feedforward = PredictiveTrackingController(
            TrackingConfig(velocity_gain=.0001, position_gain=.1))
        moving = []
        for i in range(12):
            # Arrives near center while still moving rapidly to the right.
            moving.append(observation(265+i*5, 240, 1+i*.1))
        for item in moving:
            with_decision = with_feedforward.update(item, item.timestamp, .1)
            without_decision = without_feedforward.update(item, item.timestamp, .1)
        self.assertGreater(with_decision.normalized_target_velocity[0], .1)
        self.assertGreater(with_decision.pan, without_decision.pan)

    def test_command_speed_acceleration_and_jerk_are_bounded(self):
        axis = JerkLimitedAxis(max_speed=24., max_acceleration=18., max_jerk=45.)
        previous_speed = previous_acceleration = 0.
        for desired in [24.]*60+[-24.]*120+[0.]*100:
            speed, acceleration, jerk = axis.update(desired, .05)
            self.assertLessEqual(abs(speed), 24.+1e-9)
            self.assertLessEqual(abs(acceleration), 18.+1e-9)
            self.assertLessEqual(abs(jerk), 45.+1e-7)
            self.assertLessEqual(abs((acceleration-previous_acceleration)/.05), 45.+1e-7)
            self.assertLessEqual(abs(speed-previous_speed), 18*.05+1e-7)
            previous_speed, previous_acceleration = speed, acceleration

    def test_stronger_braking_reduces_stopping_distance_without_lowering_top_speed(self):
        normal = JerkLimitedAxis(max_speed=50., max_acceleration=40., max_jerk=100.)
        braking = JerkLimitedAxis(max_speed=50., max_acceleration=40., max_jerk=100.,
                                  max_deceleration=110., max_braking_jerk=300.)
        for _ in range(30):
            normal.update(50., .05)
            braking.update(50., .05)
        self.assertAlmostEqual(normal.speed, braking.speed, delta=.01)
        for _ in range(10):
            normal.update(0., .05)
            braking.update(0., .05)
        self.assertLess(abs(braking.speed), abs(normal.speed))

    def test_stale_lost_unknown_and_track_change_are_safe(self):
        for target, now in [(None, 2.), (observation(timestamp=1.), 2.),
                            (observation(reliable=False), 1.),
                            (observation(state='STATIC'), 1.)]:
            decision = PredictiveTrackingController().update(target, now, .1)
            self.assertTrue(decision.emergency_stop)
            self.assertEqual((decision.pan, decision.tilt), (0, 0))
        controller = PredictiveTrackingController()
        for i in range(10):
            controller.update(observation(500, 240, 1+i*.1, 1), 1+i*.1, .1)
        changed = controller.update(observation(320, 240, 2.1, 2), 2.1, .1)
        self.assertEqual(changed.track_id, 2)
        self.assertLessEqual(abs(changed.pan), 1)

    def test_locked_tracking_box_does_not_require_motion_reclassification(self):
        controller = PredictiveTrackingController()
        for i in range(15):
            decision = controller.update(
                observation(500, 240, 1+i*.1, state='TRACKING'), 1+i*.1, .1)
        self.assertFalse(decision.emergency_stop)
        self.assertGreater(decision.pan, 0)

    def test_locked_static_target_enters_hysteretic_center_hold(self):
        controller = PredictiveTrackingController()
        for i in range(20):
            controller.update(observation(500, 240, 1+i*.1), 1+i*.1, .1)
        stopped = controller.update(
            observation(345, 250, 3.1, state='LOCKED_STATIC'), 3.1, .1)
        self.assertEqual((stopped.pan, stopped.tilt), (0, 0))
        self.assertEqual(stopped.reason, 'static center hold')
        for i, x in enumerate((350, 355, 360, 350)):
            held = controller.update(
                observation(x, 250, 3.2+i*.1, state='LOCKED_STATIC'), 3.2+i*.1, .1)
            self.assertEqual((held.pan, held.tilt), (0, 0))

    def test_locked_static_target_releases_hold_after_large_offset(self):
        controller = PredictiveTrackingController()
        controller.update(observation(320, 240, 1., state='LOCKED_STATIC'), 1., .1)
        for i in range(15):
            decision = controller.update(
                observation(500, 240, 1.1+i*.1, state='LOCKED_STATIC'), 1.1+i*.1, .1)
        self.assertGreater(decision.pan, 0)

    def test_invalid_configuration_and_observations_rejected(self):
        for kwargs in ({'deadband_x': 1.}, {'pan_limit': 61}, {'minimum_command': 51},
                       {'prediction_horizon': float('nan')}):
            with self.assertRaises(ValueError):
                TrackingConfig(**kwargs)
        with self.assertRaises(ValueError):
            TrackingObservation(0., 1, (0, 0, 0, 10), 640, 480)


class SafetyTests(unittest.TestCase):
    def test_position_guard_blocks_outward_but_allows_return(self):
        guard = PositionGuard({'azimuth': '1000', 'elevation': '100'},
                              {'azimuth': (0, 3600), 'elevation': (-900, 2700)}, 150, 40)
        self.assertEqual(guard.apply(-10, 10, {'azimuth': '1150', 'elevation': '250'}),
                         (0, 0, ['pan', 'tilt']))
        self.assertEqual(guard.apply(10, -10, {'azimuth': '1150', 'elevation': '250'}),
                         (10, -10, []))

    def test_continuous_guard_uses_mechanical_limits_without_start_excursion(self):
        guard = PositionGuard({'azimuth': '1000', 'elevation': '100'},
                              {'azimuth': (0, 3600), 'elevation': (-900, 2700)}, None, 40)
        self.assertEqual(guard.apply(-40, 40, {'azimuth': '1800', 'elevation': '900'}),
                         (-40, 40, []))

    def test_client_supports_two_axes_and_zero_zoom_with_bound(self):
        client = HikvisionClient(password='fake')
        with patch.object(client, 'request_xml') as request:
            client.continuous_move(12, -8, command_limit=24)
        path, method, body = request.call_args.args
        self.assertTrue(path.endswith('/continuous'))
        self.assertEqual(method, 'PUT')
        self.assertEqual([node.text for node in body], ['12', '-8', '0'])
        self.assertEqual(body.tag, 'PTZData')
        self.assertEqual(body.attrib, {})
        for values in [(25, 0), (0, -25), (True, 0)]:
            with self.assertRaises(ValueError):
                client.continuous_move(*values, command_limit=24)

    def test_watchdog_trips_and_retries_stop(self):
        class Event:
            def __init__(self, value=False): self.value = value
            def set(self): self.value = True
            def is_set(self): return self.value
            def wait(self, timeout): return self.value
        device = SimpleNamespace(ptz_status=lambda: {}, stop=Mock(
            side_effect=[CameraError('lost'), None, None, None]))
        heartbeat = SimpleNamespace(value=0., get_lock=contextlib.nullcontext)
        ready, armed, done, tripped = Event(), Event(True), Event(), Event()
        results = queue.Queue()
        with patch('ptz.target_tracker.HikvisionClient', return_value=device), \
                patch('ptz.target_tracker.time.sleep'):
            watchdog_stop_worker('192.168.1.64', 'admin', 'fake', 80, ready, armed,
                                 done, tripped, heartbeat, results, 30.)
        self.assertTrue(ready.value)
        self.assertTrue(tripped.value)
        self.assertEqual(device.stop.call_count, 4)
        self.assertEqual([results.get() for _ in range(4)], [False, True, True, True])


class TargetLockTests(unittest.TestCase):
    def test_other_salient_target_cannot_steal_active_lock(self):
        lock = PersistentTargetLock(lost_grace=2.)
        first, other = motion_target(7), motion_target(9)
        self.assertEqual(lock.update([first, other], first, 0.)[0].observation.track_id, 7)
        target, status = lock.update([first, other], other, .1)
        self.assertEqual((target.observation.track_id, status), (7, 'TRACKING'))

    def test_lock_holds_through_short_gap_then_releases(self):
        lock = PersistentTargetLock(lost_grace=2.)
        first, other = motion_target(7), motion_target(9)
        other.observation.box = (400, 300, 450, 390)
        lock.update([first], first, 0.)
        self.assertEqual(lock.update([other], other, 1.), (None, 'HOLDING'))
        self.assertEqual(lock.track_id, 7)
        self.assertEqual(lock.update([other], other, 2.1), (None, 'RELEASED'))
        self.assertIsNone(lock.track_id)

    def test_nearby_same_person_box_reassociates_changed_tracker_id(self):
        lock = PersistentTargetLock(lost_grace=2.)
        first = motion_target(7)
        changed = motion_target(19)
        changed.observation.box = (14, 10, 54, 90)
        lock.update([first], first, 0.)
        target, status = lock.update([changed], None, .1)
        self.assertEqual((target.observation.track_id, status), (19, 'REASSOCIATED'))
        self.assertEqual(lock.track_id, 19)


if __name__ == '__main__':
    unittest.main()
