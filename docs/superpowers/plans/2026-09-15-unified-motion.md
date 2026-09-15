# Unified Motion Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Detect complete moving semantic objects with one shared camera-motion estimate, persistent IDs and stable target selection.

**Architecture:** YOLO prediction feeds the installed BoT-SORT directly. A shared GMC adapter provides a partial affine warp to both tracking and object-motion classification; target selection consumes only confirmed moving observations.

**Tech Stack:** Python 3.14, NumPy, OpenCV 5, Ultralytics 8.4.144, uv, unittest.

**Spec:** docs/superpowers/specs/2026-09-15-unified-motion-design.md

## Global Constraints

- Preserve the existing MOG2 source and tests (now located in `old/`); preserve existing dependency versions. Add only the mandatory BoT-SORT assignment dependency `lap>=0.5.12,<0.6` via uv (no ReID dependencies).
- Default model `yolo26n.pt`, CPU, imgsz 416, classes `person,dog,cat`; load once and predict once per frame.
- Track allowed semantic objects before filtering motion. ReID disabled.
- Estimate GMC exactly once per frame. Use LK forward/backward consistency and RANSAC partial affine, not a full homography.
- Failed GMC uses identity for tracking, but UNKNOWN for new movement evidence; never accumulate motion/static evidence on failed GMC.
- Use measured continuous observations only for residual speed. File timestamps use source FPS; live timestamps use monotonic acquisition time.
- Never expose stale coordinates for a lost target. Short loss holds its ID, long loss unlocks.
- No physical PTZ instructions, automatic camera opening, extra model downloads or capture automation this phase.
- Q, Esc and window close release resources. Avoid leaking RTSP credentials in application error messages.

## Task 1: Shared camera motion

**Files:** Create `camera_motion.py`, `test_camera_motion.py`.

**Interfaces:** `CameraMotionResult(warp, reliable, reason, tracked_points, inlier_ratio, coverage, motion_mask, valid_mask)` where warp is float32 (2,3), masks uint8 image-size. `transform_point(point)` and `motion_ratio(box)` return a transformed center and valid-pixel ratio. `SharedCameraMotion.apply(frame, detections=None)` returns warp and updates `.last_result`; `.set_detections(boxes)` stages the complete semantic set for the next apply (BoT-SORT passes only high-confidence boxes); `.method='sparseOptFlow'`, `.calls`, `.reset_params()` match BoT-SORT GMC. xyxy boxes are full-frame coordinates. Constructor accepts `max_width=640`, `diff_threshold=25`, `min_points=12`.

- [ ] RED: create seeded textured images, initialize with previous semantic boxes, translate next frame 7px right / 4px down and assert trustworthy warp and low residual outside invalid border. Verify textureless scenes unreliable, stationary texture identity, excluded objects do not dominate, brightness offset removed, resizing warp restored to input coordinates, invalid new-view mask zero.

```python
rng = np.random.default_rng(42)
frame = rng.integers(0, 180, (240, 320, 3), dtype=np.uint8)
gmc = SharedCameraMotion()
gmc.apply(frame, np.empty((0, 4)))
shift = np.float32([[1, 0, 7], [0, 1, 4]])
gmc.apply(cv2.warpAffine(frame, shift, (320, 240)), np.empty((0, 4)))
assert gmc.last_result.reliable
np.testing.assert_allclose(gmc.last_result.warp, shift, atol=0.7)
assert not gmc.last_result.valid_mask[:, :7].any()
```

- [ ] Run `.venv\Scripts\python.exe -m unittest test_camera_motion -v`; capture expected missing module/API red output before implementation.
- [ ] Implement masked background corner extraction, LK consistency, foreground exclusion in both frames, RANSAC min points/inlier ratio/spatial coverage and finite/plausible-transform gates. Store current frame even after failure to recover next frame. Shape changes reset. Align previous gray, erode valid warped mask, estimate background median brightness offset and threshold blurred absolute residual. Never mark invalid pixels as moving. Result helpers clip boxes and handle empty overlap.
- [ ] Re-run focused tests; record red/green evidence and self-review in report. No commit: repository has no initial user history.

## Task 2: Object motion and salient lock

**Files:** Create `object_motion.py`, `test_object_motion.py`.

**Interfaces:** `ObjectObservation(track_id:int, box:tuple[float,float,float,float], label:str, confidence:float)`. `MotionConfig` accepts start_speed=0.08, stop_speed=0.04 (diagonals/s), start_ratio=0.06, stop_ratio=0.025, start_frames=3, stop_frames=5, unreliable_grace=0.5, history_ttl=2.0 (seconds). `ObjectMotionClassifier(config=None).update(observations, camera:CameraMotionResult, timestamp:float)->list[MotionTarget]`; `MotionTarget` fields observation, state (`UNKNOWN`, `STATIC`, `MOVING`), reliable, speed_px_s, normalized_speed, motion_ratio, residual:(dx,dy), moving_duration. `.histories` pruned by last_seen. `SalientTargetSelector(lost_grace=0.7).select(targets, timestamp, frame_shape)->TargetSelection` with track_id optional, target optional and status (`NONE`, `TRACKING`, `LOST`). No observation available on LOST.

- [ ] RED: assertions for moving after three reliable consecutive samples, still after five, global warp removal, internal mask motion at unchanged center, no evidence during failure/occlusion, failure grace expiration UNKNOWN, ID history TTL, normalized units, invalid overlap UNKNOWN, locked ID stable despite bigger candidate, short disappearance LOST/no target, long disappearance reselect.

```python
classifier = ObjectMotionClassifier()
for i in range(5):
    observation = ObjectObservation(1, (20+i*5, 30, 60+i*5, 90), 'person', .9)
    targets = classifier.update([observation], camera, i / 10)
assert targets[0].state == 'MOVING'
selection = SalientTargetSelector().select(targets, .4, (240, 320))
assert selection.track_id == 1 and selection.target is not None
```

- [ ] Run `.venv\Scripts\python.exe -m unittest test_object_motion -v`; expected absent module/API before implementation.
- [ ] Implement per-ID continuous-observation state: gaps clear comparison continuity/evidence; quality failure does not increment counters and eventually makes UNKNOWN; speed from raw measured center after warp divided by actual positive dt and bbox diagonal; internal valid-region ratio independent of center; hysteresis, moving duration, history prune. Score normalized area/speed/ratio/persistence among reliable MOVING, preserve lock on visible MOVING and temporary low-quality state; LOST holds ID without old target.
- [ ] Run focused tests, document red/green and self-review in report. No commits/history changes.

## Task 3: BoT-SORT application and documentation

**Files:** Create `unified_motion_detector.py`, `test_unified_motion_detector.py`; modify `README.md` only to document commands and retained baseline.

**Interfaces:** `UnifiedMotionDetector(model_path='yolo26n.pt', classes='person,dog,cat', imgsz=416, confidence=.35, motion_config=None)` owns model, tracker, shared camera, classifier, selector. `.process(frame,timestamp)->FrameResult` with targets, selection, camera. `.update_tracks(boxes, frame)` uses installed BOTSORT on real Ultralytics `Boxes.cpu().numpy()` and maps output detection index to measured box. BoT-SORT args loaded from built-in YAML, with_reid=False; shared GMC installed by assigning `.gmc`. Model prediction called once; always update tracker, including empty boxes. Unknown classes error. Selection receives frame shape.

- [ ] RED: write true Ultralytics Boxes-to-BOTSORT tests for sequential translated boxes, persistent ID and one shared GMC call per frame (including no detections), plus combined global-camera/object-moving sequence using actual tracker. Test `parse_source`, CSV no stale LOST box, file timestamp, unknown class, zero-frame resource cleanup with injected failing input. Verify local weight exists; run actual model inference on generated frames, not mocked detection output.

```python
boxes = Boxes(np.array([[30, 40, 80, 110, .9, 0]], np.float32), (240, 320))
observations = detector.update_tracks(boxes, frame)
assert detector.camera.calls == 1
assert observations[0].box == (30, 40, 80, 110)
```

- [ ] Run `.venv\Scripts\python.exe -m unittest test_unified_motion_detector -v`; expected missing application/API.
- [ ] Implement explicit prediction/tracker pipeline and CLI `--source`, `--model`, `--classes`, `--imgsz`, `--confidence`, `--resize-width`, `--show-all`, `--show-mask`, `--headless`, `--max-frames`, `--csv`, motion speed/ratio/confirmation thresholds. Validate positive dimensions, finite thresholds and nonnegative grace. Draw full-object yellow moving boxes/red selection, optional gray STATIC/UNKNOWN and status/FPS/GMC overlay. CSV rows carry timestamp, ID/label/box, state, selected ID/status, residual speed, motion ratio and GMC quality; LOST rows contain no stale coordinates. Input file timeline from FPS; live capture timestamp monotonic. Try/finally releases capture, CSV and created windows, Q/Esc/X. Suppress credential-bearing OpenCV input logs and sanitize application errors. No automatic RTSP connection in tests.
- [ ] README lead with new uv command; explain baseline `uv run python old/motion_detector.py --source 0`, metric units, no full homography, limits and no real PTZ yet.
- [ ] Run focused tests and full new suite plus baseline with `unittest discover -s old`; run CLI help and actual temporary generated video headless inference. Recheck old file SHA256.
- [ ] Independent final code review; fix Important issues with covering failing regression tests, rerun verification, hand off command.
