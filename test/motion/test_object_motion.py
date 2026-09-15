"""Synthetic pair evidence exercises the real shared-camera result API."""

import sys
import unittest
from pathlib import Path

import numpy as np

APP_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if APP_ROOT not in sys.path:  # Tests live in test/motion/ but import motion/ modules.
    sys.path.insert(0, APP_ROOT)

from motion.camera_motion import CameraMotionResult
from motion.object_motion import (
    MotionConfig, MotionTarget, ObjectMotionClassifier, ObjectObservation,
    SalientTargetSelector, TargetSelection,
)


def camera(reliable=True, warp=None, valid=None, motion=None):
    shape = (240, 320)
    return CameraMotionResult(
        np.float32([[1, 0, 0], [0, 1, 0]]) if warp is None else np.float32(warp),
        reliable, 'synthetic pair', 30, .95, .5,
        np.zeros(shape, np.uint8) if motion is None else motion,
        np.full(shape, 255, np.uint8) if valid is None else valid,
    )


def observation(x=20, track_id=1, box=None):
    return ObjectObservation(track_id, box or (x, 30, x+40, 90), 'person', .9)


def moving_target(track_id=1, box=(20, 30, 60, 90), **changes):
    fields = dict(observation=observation(track_id=track_id, box=box),
                  state='MOVING', reliable=True, speed_px_s=20.,
                  normalized_speed=.3, motion_ratio=.1, residual=(2., 0.),
                  moving_duration=1.)
    fields.update(changes)
    return MotionTarget(**fields)


class ObjectMotionTests(unittest.TestCase):
    def setUp(self):
        self.classifier = ObjectMotionClassifier()
        self.camera = camera()

    def sample(self, x, time, cam=None):
        return self.classifier.update([observation(x)], cam or self.camera, time)[0]

    def establish_moving(self):
        for i in range(4):
            result = self.sample(20+5*i, i*.1)
        self.assertEqual(result.state, 'MOVING')
        return result

    def test_three_consecutive_reliable_pairs_confirm_moving(self):
        results = [self.sample(20+5*i, i*.1) for i in range(4)]
        self.assertEqual([t.state for t in results], ['UNKNOWN']*3 + ['MOVING'])
        self.assertFalse(results[0].reliable)
        self.assertTrue(all(t.reliable for t in results[1:]))

    def test_five_still_pairs_confirm_static(self):
        results = [self.sample(20, i*.1) for i in range(6)]
        self.assertEqual([t.state for t in results], ['UNKNOWN']*5 + ['STATIC'])

    def test_moving_requires_five_still_pairs_to_stop(self):
        self.establish_moving()
        results = [self.sample(35, .4+i*.1) for i in range(5)]
        self.assertEqual([t.state for t in results], ['MOVING']*4 + ['STATIC'])
        self.assertEqual(results[-1].moving_duration, 0.)

    def test_translation_is_removed_from_raw_center(self):
        self.sample(20, 0.)
        shifted = camera(warp=[[1, 0, 5], [0, 1, 3]])
        for i in range(1, 6):
            target = self.classifier.update([
                observation(box=(20+5*i, 30+3*i, 60+5*i, 90+3*i))], shifted, i*.1)[0]
            self.assertEqual(target.residual, (0., 0.))
            self.assertEqual(target.speed_px_s, 0.)
        self.assertEqual(target.state, 'STATIC')

    def test_rotation_and_scale_remove_camera_motion(self):
        self.sample(20, 0.)  # Previous center is (40, 60).
        result = self.classifier.update([observation(box=(28, 26, 68, 86))],
            camera(warp=[[0, -1.125, 115.5], [1.125, 0, 11]]), .1)[0]
        self.assertEqual(result.residual, (0., 0.))
        self.assertEqual(result.normalized_speed, 0.)

    def test_speed_uses_actual_dt_and_box_diagonal(self):
        self.sample(20, 1.)
        target = self.sample(26, 1.2)
        self.assertEqual(target.residual, (6., 0.))
        self.assertAlmostEqual(target.speed_px_s, 30.)
        self.assertAlmostEqual(target.normalized_speed, .4160251471689218)

    def test_normalized_speed_is_resolution_independent(self):
        targets = []
        for scale in (1, 2):
            classifier = ObjectMotionClassifier()
            for i in range(2):
                box = tuple(scale*v for v in (20+6*i, 30, 60+6*i, 90))
                target = classifier.update([observation(box=box)], self.camera, i*.2)[0]
            targets.append(target)
        self.assertAlmostEqual(targets[0].normalized_speed, targets[1].normalized_speed)
        self.assertAlmostEqual(targets[1].speed_px_s, 60.)

    def test_internal_motion_confirms_unchanged_center(self):
        motion = np.zeros((240, 320), np.uint8)
        motion[30:90, 20:24] = 255  # 240 / 2400 = .1.
        internal = camera(motion=motion)
        for i in range(4):
            target = self.sample(20, i*.1, internal)
        self.assertEqual(target.state, 'MOVING')
        self.assertEqual(target.speed_px_s, 0.)
        self.assertAlmostEqual(target.motion_ratio, .1)

    def test_ratio_counts_only_valid_pixels(self):
        valid = np.zeros((240, 320), np.uint8)
        valid[30:90, 20:40] = 255
        motion = np.zeros_like(valid)
        motion[30:90, 20:22] = 255
        motion[30:90, 40:60] = 255  # Invalid changes must not count.
        self.sample(20, 0.)
        target = self.sample(20, .1, camera(valid=valid, motion=motion))
        self.assertTrue(target.reliable)
        self.assertAlmostEqual(target.motion_ratio, .1)

    def test_zero_or_tiny_valid_overlap_cannot_confirm_static(self):
        for pixels in (0, 1, 19):
            with self.subTest(valid_columns=pixels):
                self.classifier = ObjectMotionClassifier()
                valid = np.zeros((240, 320), np.uint8)
                valid[30:90, 20:20+pixels] = 255
                for i in range(7):
                    target = self.sample(20, i*.1, camera(valid=valid))
                self.assertEqual(target.state, 'UNKNOWN')
                self.assertFalse(target.reliable)

    def test_tiny_box_and_mostly_offscreen_box_lack_valid_support(self):
        for box in ((20, 30, 22, 32), (-100, 30, 20, 90)):
            with self.subTest(box=box):
                classifier = ObjectMotionClassifier()
                for i in range(7):
                    target = classifier.update([observation(box=box)], self.camera, i*.1)[0]
                self.assertEqual(target.state, 'UNKNOWN')
                self.assertFalse(target.reliable)

    def test_failure_breaks_start_confirmation_and_recovery_uses_latest_center(self):
        for i in range(3):
            self.sample(20+5*i, i*.1)
        failed = self.sample(100, .3, camera(False))
        self.assertEqual(failed.state, 'UNKNOWN')
        self.assertFalse(failed.reliable)
        self.assertEqual(failed.speed_px_s, 0.)
        results = [self.sample(105+5*i, .4+i*.1) for i in range(3)]
        self.assertEqual([t.state for t in results], ['UNKNOWN', 'UNKNOWN', 'MOVING'])
        self.assertAlmostEqual(results[0].speed_px_s, 50.)

    def test_failure_holds_short_then_expires_unknown_without_evidence(self):
        self.establish_moving()
        short = self.sample(35, .5, camera(False))
        long = self.sample(35, .9, camera(False))
        self.assertEqual(short.state, 'MOVING')
        self.assertFalse(short.reliable)
        self.assertEqual(short.moving_duration, 0.)
        self.assertEqual(long.state, 'UNKNOWN')
        self.assertFalse(long.reliable)
        self.assertEqual(long.moving_duration, 0.)

    def test_failure_cannot_accumulate_stop_evidence(self):
        self.establish_moving()
        for i in range(4):
            self.sample(35, .4+i*.01)
        self.sample(35, .45, camera(False))
        results = [self.sample(35, .5+i*.01) for i in range(5)]
        self.assertEqual([t.state for t in results], ['MOVING']*4+['STATIC'])

    def test_long_failure_gap_expires_before_reliable_recovery(self):
        self.establish_moving()
        self.sample(40, .4, camera(False))
        recovered = self.sample(45, 1.)
        self.assertTrue(recovered.reliable)
        self.assertEqual(recovered.state, 'UNKNOWN')
        self.assertEqual(recovered.moving_duration, 0.)

    def test_occlusion_breaks_confirmation_and_cannot_use_stale_speed(self):
        for i in range(3):
            self.sample(20+5*i, i*.1)
        self.assertEqual(self.classifier.update([], self.camera, .3), [])
        returned = self.sample(100, .4)
        self.assertEqual(returned.state, 'UNKNOWN')
        self.assertFalse(returned.reliable)
        self.assertEqual(returned.residual, (0., 0.))
        results = [self.sample(105+5*i, .5+i*.1) for i in range(3)]
        self.assertEqual([t.state for t in results], ['UNKNOWN', 'UNKNOWN', 'MOVING'])
        self.assertAlmostEqual(results[0].speed_px_s, 50.)

    def test_nonpositive_dt_is_not_evidence(self):
        for timestamp in (0., -.1):
            with self.subTest(timestamp=timestamp):
                self.classifier = ObjectMotionClassifier()
                self.sample(20, 0.)
                target = self.sample(100, timestamp)
                self.assertFalse(target.reliable)
                self.assertEqual(target.speed_px_s, 0.)
                self.assertEqual(target.state, 'UNKNOWN')

    def test_deadband_breaks_confirmation_and_preserves_existing_state(self):
        for i in range(3):
            self.sample(20+5*i, i*.1)
        # 0.5 px / .1 s / sqrt(40**2+60**2) = .0693, in deadband.
        self.sample(30.5, .3)
        self.assertEqual(self.sample(35.5, .4).state, 'UNKNOWN')
        self.sample(40.5, .5)
        self.assertEqual(self.sample(45.5, .6).state, 'MOVING')
        self.assertEqual(self.sample(46, .7).state, 'MOVING')

    def test_moving_duration_counts_only_reliable_moving_time(self):
        self.establish_moving()
        target = self.sample(40, .4)
        self.assertAlmostEqual(target.moving_duration, .1)
        failed = self.sample(45, .5, camera(False))
        self.assertAlmostEqual(failed.moving_duration, .1)
        recovered = self.sample(50, .6)
        self.assertAlmostEqual(recovered.moving_duration, .1)
        self.assertAlmostEqual(self.sample(55, .7).moving_duration, .2)

    def test_histories_prune_by_last_seen_and_visible_ids_refresh(self):
        self.sample(20, 0.)
        self.classifier.update([observation(track_id=2)], self.camera, 1.)
        self.classifier.update([], self.camera, 2.)
        self.assertIn(1, self.classifier.histories)
        self.classifier.update([observation(track_id=2)], self.camera, 2.01)
        self.assertNotIn(1, self.classifier.histories)
        self.assertIn(2, self.classifier.histories)
        self.classifier.update([], self.camera, 4.02)
        self.assertEqual(self.classifier.histories, {})

    def test_id_evidence_is_independent(self):
        for i in range(6):
            targets = self.classifier.update([observation(20+i*5), observation(150, 2)],
                                             self.camera, i*.1)
        self.assertEqual([t.state for t in targets], ['MOVING', 'STATIC'])

    def test_invalid_box_cannot_seed_comparison(self):
        for box in ((20, 30, 20, 90), (20, 30, float('nan'), 90)):
            with self.subTest(box=box):
                classifier = ObjectMotionClassifier()
                target = classifier.update([observation(box=box)], self.camera, 0.)[0]
                self.assertFalse(target.reliable)
                returned = classifier.update([observation(100)], self.camera, .1)[0]
                self.assertFalse(returned.reliable)
                self.assertEqual(returned.speed_px_s, 0.)

    def test_config_rejects_invalid_thresholds_and_counts(self):
        for name in ('start_speed', 'stop_speed', 'start_ratio', 'stop_ratio',
                     'unreliable_grace', 'history_ttl'):
            for value in (-1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    MotionConfig(**{name: value})
        for name in ('start_frames', 'stop_frames'):
            for value in (0, -1, 1.5, float('nan'), float('inf'), True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    MotionConfig(**{name: value})
        for fields in (dict(stop_speed=.08), dict(stop_speed=.09),
                       dict(stop_ratio=.06), dict(stop_ratio=.07)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                MotionConfig(**fields)

    def test_single_confirmation_and_zero_grace_are_supported(self):
        classifier = ObjectMotionClassifier(MotionConfig(start_frames=1, stop_frames=1,
                                                        unreliable_grace=0., history_ttl=0.))
        classifier.update([observation()], self.camera, 0.)
        target = classifier.update([observation(25)], self.camera, .1)[0]
        # Zero TTL expires even visible history if last_seen is in the past.
        self.assertEqual(target.state, 'UNKNOWN')
        classifier = ObjectMotionClassifier(MotionConfig(start_frames=1, stop_frames=1,
                                                        unreliable_grace=0.))
        classifier.update([observation()], self.camera, 0.)
        self.assertEqual(classifier.update([observation(25)], self.camera, .1)[0].state, 'MOVING')
        self.assertEqual(classifier.update([observation(25)], camera(False), .2)[0].state, 'UNKNOWN')


class SalientTargetTests(unittest.TestCase):
    def setUp(self):
        self.selector = SalientTargetSelector()
        self.shape = (240, 320, 3)

    def select(self, targets, time):
        return self.selector.select(targets, time, self.shape)

    def test_only_reliable_moving_candidates_can_acquire_lock(self):
        result = self.select([moving_target(state='STATIC'),
                              moving_target(2, reliable=False),
                              moving_target(3, state='UNKNOWN')], 0.)
        self.assertIsInstance(result, TargetSelection)
        self.assertEqual(result.status, 'NONE')
        self.assertIsNone(result.track_id)
        self.assertIsNone(result.target)

    def test_score_combines_area_speed_ratio_and_persistence(self):
        for changes in (dict(box=(20, 30, 100, 150)), dict(normalized_speed=2.),
                        dict(motion_ratio=.8), dict(moving_duration=5.)):
            with self.subTest(changes=changes):
                selector = SalientTargetSelector()
                targets = [moving_target(1), moving_target(2, **changes)]
                self.assertEqual(selector.select(targets, 0., self.shape).track_id, 2)

    def test_lock_is_stable_despite_bigger_faster_candidate(self):
        first = moving_target()
        self.assertEqual(self.select([first], 0.).track_id, 1)
        current = moving_target(box=(25, 30, 65, 90))
        selection = self.select([moving_target(2, (0, 0, 300, 230), normalized_speed=3.),
                                 current], .1)
        self.assertEqual(selection.status, 'TRACKING')
        self.assertEqual(selection.track_id, 1)
        self.assertIs(selection.target, current)

    def test_short_disappearance_lost_has_no_stale_target(self):
        self.select([moving_target()], 0.)
        lost = self.select([moving_target(2)], .2)
        self.assertEqual(lost.status, 'LOST')
        self.assertEqual(lost.track_id, 1)
        self.assertIsNone(lost.target)
        lost_again = self.select([], .6)
        self.assertEqual(lost_again.status, 'LOST')
        self.assertIsNone(lost_again.target)

    def test_return_before_grace_keeps_id_and_current_observation(self):
        self.select([moving_target()], 0.)
        self.select([], .2)
        current = moving_target(box=(100, 30, 140, 90), reliable=False)
        selected = self.select([current, moving_target(2)], .3)
        self.assertEqual(selected.track_id, 1)
        self.assertIs(selected.target, current)

    def test_long_disappearance_reselects_or_returns_none(self):
        self.select([moving_target()], 0.)
        self.select([], .2)
        selected = self.select([moving_target(2)], .71)
        self.assertEqual(selected.track_id, 2)
        self.assertEqual(selected.status, 'TRACKING')
        self.select([], .8)
        none = self.select([], 1.42)
        self.assertEqual(none.status, 'NONE')
        self.assertIsNone(none.track_id)
        self.assertIsNone(none.target)

    def test_confirmed_stop_releases_lock_immediately(self):
        self.select([moving_target()], 0.)
        selected = self.select([moving_target(state='STATIC'), moving_target(2)], .1)
        self.assertEqual(selected.track_id, 2)

    def test_expired_unknown_releases_lock_inside_selector_grace(self):
        for alternative in (None, moving_target(2)):
            with self.subTest(alternative=alternative is not None):
                self.selector = SalientTargetSelector()
                self.select([moving_target()], 0.)
                low = moving_target(reliable=False)
                held = self.select([low], .2)
                self.assertEqual(held.track_id, 1)
                self.assertIs(held.target, low)  # Still the current MOVING observation.
                expired = moving_target(state='UNKNOWN', reliable=False)
                targets = [expired] if alternative is None else [expired, alternative]
                selected = self.select(targets, .51)  # Quality .5 s < selector .7 s.
                self.assertEqual(selected.track_id, None if alternative is None else 2)
                self.assertEqual(selected.status, 'NONE' if alternative is None else 'TRACKING')
                self.assertIs(selected.target, alternative)

    def test_low_quality_moving_does_not_extend_hold_forever(self):
        self.select([moving_target()], 0.)
        low = moving_target(reliable=False)
        self.assertEqual(self.select([low], .2).track_id, 1)
        self.assertEqual(self.select([low, moving_target(2)], .71).track_id, 2)

    def test_selector_grace_validation(self):
        for value in (-1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SalientTargetSelector(lost_grace=value)
        selector = SalientTargetSelector(lost_grace=0.)
        selector.select([moving_target()], 0., self.shape)
        self.assertEqual(selector.select([], .01, self.shape).status, 'NONE')


if __name__ == '__main__':
    unittest.main()
