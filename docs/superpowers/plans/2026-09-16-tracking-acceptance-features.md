# 运动目标跟踪验收功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有实体球机跟踪程序中加入预测越界报警、标注录像、实时指标、自适应预测、连续跟踪验收、实时误差曲线和单张清晰抓拍。

**Architecture:** 保留 `ball_camera_track.py` 的检测与 PTZ 主链，把报警、录像、指标、会话文件、自适应和 UI 绘制拆到独立 `tracking/` 包。各模块使用纯数据接口，输出故障不得阻塞 PTZ 控制，控制故障仍走现有看门狗和重复停止路径。

**Tech Stack:** Python 3.14、OpenCV、NumPy、Ultralytics YOLO/BoT-SORT、`unittest`、Hikvision ISAPI。

**Spec:** `docs/superpowers/specs/2026-09-16-tracking-acceptance-features-design.md`

## Global Constraints

- 继续使用 OpenCV 窗口，不引入 Qt 或新的大型依赖。
- 默认离线模式不得连接相机；只有 `--execute` 才允许 PTZ 输出。
- `--continuous` 只取消时间和相对起点行程限制，机械边界、观测超时、速度上限和独立看门狗必须保留。
- 边界报警预测时间固定默认 0.5 秒、危险区 8%、同方向冷却 3 秒。
- 卡尔曼正常控制预测保持 0.1 秒；急转弯时临时降为 0 秒。
- 稳定跟踪要求同一物理目标连续有效 10 秒；缺失立即重置当前连续计时，达标结果锁存。
- 每次运行最多保存一张 `center-capture.jpg`，且必须是未标注原图。
- 录像默认启用，文件名 `annotated.mp4`，默认 10 FPS；`--no-record` 显式关闭。
- 输出和报告不得包含密码、带凭据的 RTSP URL 或 `.env` 内容。
- 输出附加功能失败是非致命故障；相机、控制或看门狗故障必须立即停止 PTZ。
- 不把 ISAPI 指令值宣称为已标定角速度，不把软件 jerk 限制宣称为电机物理 jerk 保证。

---

### Task 1: 会话文件与实时指标

**Files:**
- Create: `tracking/__init__.py`
- Create: `tracking/session.py`
- Create: `tracking/metrics.py`
- Create: `test/tracking/__init__.py`
- Create: `test/tracking/test_session.py`
- Create: `test/tracking/test_metrics.py`

**Interfaces:**
- Produces: `TrackingSession(directory: Path)`, `.append_error(row: dict)`, `.append_event(event: dict)`, `.save_capture(frame) -> bool`, `.write_report(report: dict)`, `.close()`.
- Produces: `TrackingMetrics(required_seconds=10.0)`, `.update_frame(acquired_at, processed_at)`, `.update_target(timestamp, physical_key, error_x, error_y)`, `.target_missing()`, `.record_alarm()`, `.snapshot() -> MetricsSnapshot`.
- `physical_key` 是持久目标锁确认的物理身份代号；ID 重关联时保持不变，真正释放后更换。

- [ ] **Step 1: 写会话文件失败测试**

```python
# test/tracking/test_session.py
import csv, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tracking.session import TrackingSession

class SessionTests(unittest.TestCase):
    def test_streams_rows_saves_only_one_capture_and_atomically_writes_report(self):
        with tempfile.TemporaryDirectory() as root:
            session = TrackingSession(Path(root))
            session.append_error({'time': 0.1, 'error_x_px': 3., 'error_y_px': -2., 'radial_error_px': 3.6})
            session.append_event({'time': 0.1, 'type': 'BOUNDARY', 'direction': 'RIGHT', 'track_id': 7})
            frame = np.full((40, 60, 3), 127, np.uint8)
            self.assertTrue(session.save_capture(frame))
            self.assertFalse(session.save_capture(frame))
            session.write_report({'stable_tracking_achieved': True})
            session.close()
            self.assertEqual(len(list(csv.DictReader((Path(root)/'tracking-error.csv').open(encoding='utf-8-sig')))), 1)
            self.assertEqual(len(list(csv.DictReader((Path(root)/'events.csv').open(encoding='utf-8-sig')))), 1)
            self.assertTrue((Path(root)/'center-capture.jpg').is_file())
            self.assertTrue(json.loads((Path(root)/'report.json').read_text())['stable_tracking_achieved'])
            self.assertFalse((Path(root)/'report.json.tmp').exists())

    def test_optional_output_failure_is_recorded_without_raising(self):
        with tempfile.TemporaryDirectory() as root, patch('tracking.session.cv2.imwrite', return_value=False):
            session = TrackingSession(Path(root))
            self.assertFalse(session.save_capture(np.zeros((10, 10, 3), np.uint8)))
            self.assertIn('center-capture.jpg', session.output_failures)
            session.close()
```

- [ ] **Step 2: 写指标失败测试**

```python
# test/tracking/test_metrics.py
import unittest
from tracking.metrics import TrackingMetrics

class MetricsTests(unittest.TestCase):
    def test_continuous_tracking_resets_on_gap_but_pass_latches(self):
        metrics = TrackingMetrics(required_seconds=10.)
        for second in range(11):
            metrics.update_target(float(second), 'physical-1', 10., -5.)
        self.assertTrue(metrics.snapshot().stable_tracking_achieved)
        metrics.target_missing()
        self.assertEqual(metrics.snapshot().current_stable_seconds, 0.)
        self.assertTrue(metrics.snapshot().stable_tracking_achieved)

    def test_frame_age_fps_error_percentiles_and_reassociation_are_reported(self):
        metrics = TrackingMetrics(required_seconds=10.)
        for i in range(20):
            metrics.update_frame(10+i*.1, 10.03+i*.1)
            metrics.update_target(i*.1, 'physical-1', float(i), -float(i))
        metrics.record_reassociation()
        metrics.record_alarm()
        snap = metrics.snapshot()
        self.assertAlmostEqual(snap.processing_fps, 10., delta=.5)
        self.assertAlmostEqual(snap.p95_frame_age_ms, 30., delta=.1)
        self.assertEqual((snap.reassociations, snap.alarms), (1, 1))
        self.assertGreater(snap.p95_radial_error_px, 20.)
```

- [ ] **Step 3: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_session test.tracking.test_metrics -v`

Expected: FAIL，提示 `No module named 'tracking'`。

- [ ] **Step 4: 实现会话文件接口**

```python
# tracking/session.py 核心结构
class TrackingSession:
    ERROR_FIELDS = ('time', 'track_id', 'error_x_px', 'error_y_px', 'radial_error_px')
    EVENT_FIELDS = ('time', 'type', 'direction', 'track_id', 'predicted_box', 'error_x_px', 'error_y_px')

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.output_failures = []
        self.capture_saved = False
        self._error_stream, self._error_writer = self._open_csv('tracking-error.csv', self.ERROR_FIELDS)
        self._event_stream, self._event_writer = self._open_csv('events.csv', self.EVENT_FIELDS)

    def save_capture(self, frame):
        if self.capture_saved:
            return False
        ok = bool(cv2.imwrite(str(self.directory/'center-capture.jpg'), frame))
        self.capture_saved = ok
        if not ok:
            self.output_failures.append('center-capture.jpg')
        return ok

    def write_report(self, report):
        temporary = self.directory/'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.directory/'report.json')
```

CSV writer 每次写行后调用 `flush()`；`close()` 幂等关闭两个流。`append_event()` 对缺失字段补空字符串，不允许额外字段泄漏到 CSV。

- [ ] **Step 5: 实现指标接口**

```python
# tracking/metrics.py 公共类型
@dataclass(frozen=True)
class MetricsSnapshot:
    processing_fps: float
    current_frame_age_ms: float
    p95_frame_age_ms: float
    current_stable_seconds: float
    longest_stable_seconds: float
    stable_tracking_achieved: bool
    p50_radial_error_px: float
    p95_radial_error_px: float
    target_losses: int
    reassociations: int
    alarms: int

class TrackingMetrics:
    def update_target(self, timestamp, physical_key, error_x, error_y):
        if self._key != physical_key:
            self._key, self._last_target_time, self._current = physical_key, timestamp, 0.
        elif self._last_target_time is not None:
            self._current += max(0., min(.5, timestamp-self._last_target_time))
        self._last_target_time = timestamp
        self._longest = max(self._longest, self._current)
        self._passed |= self._current >= self.required_seconds
        self._radial.append(math.hypot(error_x, error_y))
```

FPS 使用最近 60 个 `processed_at` 的时间跨度；帧龄为 `max(0, processed_at-acquired_at)`；分位数使用 `numpy.percentile`，空样本返回 0。

- [ ] **Step 6: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_session test.tracking.test_metrics -v`

Expected: 4 tests PASS。

```powershell
git add tracking/__init__.py tracking/session.py tracking/metrics.py test/tracking
git commit -m "feat: add tracking session metrics"
```

---

### Task 2: 预测边界报警状态机

**Files:**
- Create: `tracking/alarm.py`
- Create: `test/tracking/test_alarm.py`

**Interfaces:**
- Consumes: 当前 `box: tuple[float, float, float, float]`、卡尔曼像素速度、帧宽高、时间戳和目标 ID。
- Produces: `BoundaryAlarm.update(timestamp: float, track_id: int | None, box, velocity, width: int, height: int) -> AlarmEvent | None`。
- Produces: `AlarmEvent(timestamp, track_id, directions, predicted_box)`；`directions` 是稳定排序的元组，顺序为 LEFT、RIGHT、TOP、BOTTOM。

- [ ] **Step 1: 写四方向、角落、冷却和丢失测试**

```python
# test/tracking/test_alarm.py
import unittest
from tracking.alarm import BoundaryAlarm, BoundaryAlarmConfig

class AlarmTests(unittest.TestCase):
    def setUp(self):
        self.alarm = BoundaryAlarm(BoundaryAlarmConfig(horizon=.5, margin_ratio=.08, cooldown=3.))

    def test_predicts_right_boundary_half_second_ahead(self):
        event = self.alarm.update(0., 7, (500, 100, 580, 300), (100., 0.), 640, 480)
        self.assertEqual(event.directions, ('RIGHT',))
        self.assertEqual(event.predicted_box, (550., 100., 630., 300.))

    def test_corner_reports_two_stably_ordered_directions(self):
        event = self.alarm.update(0., 7, (5, 5, 80, 100), (-20., -20.), 640, 480)
        self.assertEqual(event.directions, ('LEFT', 'TOP'))

    def test_same_direction_cools_down_until_safe_rearm(self):
        self.assertIsNotNone(self.alarm.update(0., 7, (550, 100, 630, 300), (0., 0.), 640, 480))
        self.assertIsNone(self.alarm.update(1., 7, (550, 100, 630, 300), (0., 0.), 640, 480))
        self.assertIsNone(self.alarm.update(4., 7, (550, 100, 630, 300), (0., 0.), 640, 480))
        self.assertIsNone(self.alarm.update(4.1, 7, (200, 100, 300, 300), (0., 0.), 640, 480))
        self.assertIsNotNone(self.alarm.update(4.2, 7, (550, 100, 630, 300), (0., 0.), 640, 480))

    def test_missing_observation_clears_prediction_without_alarm(self):
        self.assertIsNone(self.alarm.update(0., None, None, None, 640, 480))
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_alarm -v`

Expected: FAIL，提示 `tracking.alarm` 不存在。

- [ ] **Step 3: 实现纯状态机**

```python
# tracking/alarm.py 核心算法
@dataclass(frozen=True)
class BoundaryAlarmConfig:
    horizon: float = .5
    margin_ratio: float = .08
    cooldown: float = 3.

@dataclass(frozen=True)
class AlarmEvent:
    timestamp: float
    track_id: int
    directions: tuple[str, ...]
    predicted_box: tuple[float, float, float, float]

def _predict(box, velocity, horizon):
    dx, dy = velocity[0]*horizon, velocity[1]*horizon
    return box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy

class BoundaryAlarm:
    def update(self, timestamp, track_id, box, velocity, width, height):
        if track_id is None or box is None or velocity is None:
            return None
        predicted = _predict(box, velocity, self.config.horizon)
        mx, my = width*self.config.margin_ratio, height*self.config.margin_ratio
        active = tuple(name for name, hit in (
            ('LEFT', predicted[0] <= mx), ('RIGHT', predicted[2] >= width-mx),
            ('TOP', predicted[1] <= my), ('BOTTOM', predicted[3] >= height-my)) if hit)
        # 安全区方向从 armed 集合移除；仍在危险区的方向必须先返回安全区才能重报。
```

配置构造时拒绝非有限值、`horizon <= 0`、`cooldown <= 0` 和不在 `(0, .5)` 的 `margin_ratio`。状态按方向记录 `armed` 与上次触发时间；冷却期结束但未返回安全区仍不重复报警。

- [ ] **Step 4: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_alarm -v`

Expected: 4 tests PASS。

```powershell
git add tracking/alarm.py test/tracking/test_alarm.py
git commit -m "feat: add predicted boundary alarms"
```

---

### Task 3: 标注视频录像器

**Files:**
- Create: `tracking/recorder.py`
- Create: `test/tracking/test_recorder.py`

**Interfaces:**
- Produces: `AnnotatedVideoRecorder(path: Path, frame_size: tuple[int, int], fps=10.0, enabled=True)`。
- Produces: `.write(frame) -> bool`、`.close()`、只读 `.opened`、`.frames_written`、`.failure`。

- [ ] **Step 1: 写打开、尺寸校验、失败降级和幂等关闭测试**

```python
# test/tracking/test_recorder.py
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from tracking.recorder import AnnotatedVideoRecorder

class RecorderTests(unittest.TestCase):
    @patch('tracking.recorder.cv2.VideoWriter')
    def test_writes_exact_size_and_closes_once(self, factory):
        writer = Mock(); writer.isOpened.return_value = True; factory.return_value = writer
        recorder = AnnotatedVideoRecorder(Path('annotated.mp4'), (640, 480), 10.)
        self.assertTrue(recorder.write(np.zeros((480, 640, 3), np.uint8)))
        with self.assertRaises(ValueError):
            recorder.write(np.zeros((240, 320, 3), np.uint8))
        recorder.close(); recorder.close()
        self.assertEqual(writer.release.call_count, 1)
        self.assertEqual(recorder.frames_written, 1)

    @patch('tracking.recorder.cv2.VideoWriter')
    def test_encoder_failure_is_nonfatal_and_reported(self, factory):
        writer = Mock(); writer.isOpened.return_value = False; factory.return_value = writer
        recorder = AnnotatedVideoRecorder(Path('annotated.mp4'), (640, 480), 10.)
        self.assertFalse(recorder.write(np.zeros((480, 640, 3), np.uint8)))
        self.assertIn('encoder', recorder.failure.lower())

    def test_disabled_recorder_never_constructs_writer(self):
        with patch('tracking.recorder.cv2.VideoWriter') as factory:
            recorder = AnnotatedVideoRecorder(Path('annotated.mp4'), (640, 480), enabled=False)
            self.assertFalse(recorder.write(np.zeros((480, 640, 3), np.uint8)))
            factory.assert_not_called()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_recorder -v`

Expected: FAIL，提示 `tracking.recorder` 不存在。

- [ ] **Step 3: 实现录像器**

```python
# tracking/recorder.py 核心结构
class AnnotatedVideoRecorder:
    def __init__(self, path, frame_size, fps=10., enabled=True):
        self.path, self.frame_size, self.fps = Path(path), tuple(frame_size), float(fps)
        self.frames_written, self.failure, self._closed = 0, None, False
        self._writer = None
        if enabled:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.frame_size)
            if writer.isOpened():
                self._writer = writer
            else:
                writer.release()
                self.failure = 'MP4 encoder failed to open'

    def write(self, frame):
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            raise ValueError('Annotated frame size changed during recording')
        if self._writer is None:
            return False
        self._writer.write(frame)
        self.frames_written += 1
        return True
```

`close()` 以 `_closed` 防重入；录像失败只设置 `failure`，不抛出到 PTZ 主循环。

- [ ] **Step 4: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_recorder -v`

Expected: 3 tests PASS。

```powershell
git add tracking/recorder.py test/tracking/test_recorder.py
git commit -m "feat: record annotated tracking video"
```

---

### Task 4: 不规则运动与低照度自适应

**Files:**
- Create: `tracking/adaptation.py`
- Create: `test/tracking/test_adaptation.py`
- Modify: `ptz/target_tracker.py`
- Modify: `ptz/test_target_tracker.py`

**Interfaces:**
- Produces: `AdaptivePrediction(base_horizon=.1, innovation_threshold=.18, recovery_frames=5)`，`.update(measured_center, predicted_center, frame_size) -> float` 返回本帧预测时长。
- Produces: `LowLightEnhancer(threshold=55., clip_limit=2.0)`，`.apply(frame) -> tuple[np.ndarray, float, bool]`。
- Consumes: `PredictiveTrackingController.update(observation, now: float, dt: float, prediction_horizon_override: float | None = None)`；覆盖值只影响当前帧，不修改全局配置。

- [ ] **Step 1: 写急转弯和恢复测试**

```python
# test/tracking/test_adaptation.py
import unittest
import numpy as np
from tracking.adaptation import AdaptivePrediction, LowLightEnhancer

class AdaptationTests(unittest.TestCase):
    def test_large_normalized_innovation_disables_prediction_then_recovers(self):
        adaptive = AdaptivePrediction(base_horizon=.1, innovation_threshold=.18, recovery_frames=3)
        self.assertEqual(adaptive.update((500, 240), (320, 240), (640, 480)), 0.)
        self.assertEqual(adaptive.update((321, 240), (320, 240), (640, 480)), 0.)
        self.assertEqual(adaptive.update((321, 240), (320, 240), (640, 480)), 0.)
        self.assertEqual(adaptive.update((321, 240), (320, 240), (640, 480)), .1)

    def test_normal_light_is_unchanged_and_dark_frame_is_enhanced(self):
        enhancer = LowLightEnhancer(threshold=55., clip_limit=2.)
        normal = np.full((80, 100, 3), 120, np.uint8)
        output, brightness, used = enhancer.apply(normal)
        self.assertFalse(used); self.assertTrue(np.array_equal(output, normal))
        dark = np.tile(np.arange(100, dtype=np.uint8), (80, 1))//4
        dark = np.repeat(dark[:, :, None], 3, axis=2)
        output, brightness, used = enhancer.apply(dark)
        self.assertTrue(used); self.assertGreater(output.std(), dark.std())
```

- [ ] **Step 2: 写控制器预测覆盖测试**

```python
# ptz/test_target_tracker.py 新增
def test_per_frame_horizon_override_does_not_mutate_configuration(self):
    controller = PredictiveTrackingController(TrackingConfig(prediction_horizon=.1))
    for i in range(8):
        decision = controller.update(observation(300+i*5, 240, 1+i*.1), 1+i*.1, .1,
                                     prediction_horizon_override=0.)
    self.assertEqual(controller.config.prediction_horizon, .1)
    self.assertAlmostEqual(decision.predicted_center[0], observation(335, 240).center[0], delta=8.)
```

- [ ] **Step 3: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_adaptation ptz.test_target_tracker -v`

Expected: FAIL，缺少适应模块且控制器不接受覆盖参数。

- [ ] **Step 4: 实现适应模块与控制器覆盖**

```python
# tracking/adaptation.py 核心逻辑
class AdaptivePrediction:
    def update(self, measured_center, predicted_center, frame_size):
        nx = (measured_center[0]-predicted_center[0])/(frame_size[0]/2)
        ny = (measured_center[1]-predicted_center[1])/(frame_size[1]/2)
        if math.hypot(nx, ny) >= self.innovation_threshold:
            self._stable = 0
            return 0.
        self._stable += 1
        return self.base_horizon if self._stable >= self.recovery_frames else 0.

class LowLightEnhancer:
    def apply(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        brightness = float(lab[:, :, 0].mean())
        if brightness >= self.threshold:
            return frame, brightness, False
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), brightness, True
```

在控制器中使用 `horizon = config.prediction_horizon if override is None else override`，并验证覆盖值有限且位于 `[0, .5]`。

- [ ] **Step 5: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_adaptation ptz.test_target_tracker -v`

Expected: 全部 PASS。

```powershell
git add tracking/adaptation.py test/tracking/test_adaptation.py ptz/target_tracker.py ptz/test_target_tracker.py
git commit -m "feat: adapt prediction and low light input"
```

---

### Task 5: 可测试的 UI 叠加层

**Files:**
- Create: `tracking/overlay.py`
- Create: `test/tracking/test_overlay.py`

**Interfaces:**
- Produces: `OverlayState` 数据类，包含目标、控制、指标、报警、录像和抓拍状态。
- Produces: `draw_tracking_overlay(frame, state, trajectory) -> np.ndarray`，不修改输入帧。
- Produces: `update_trajectory(points, center, max_points=50)`；目标释放时调用者传 `center=None` 清空。

- [ ] **Step 1: 写绘制和轨迹上限测试**

```python
# test/tracking/test_overlay.py
import unittest
import numpy as np
from tracking.overlay import OverlayState, draw_tracking_overlay, update_trajectory

class OverlayTests(unittest.TestCase):
    def test_overlay_does_not_mutate_source_and_alarm_draws_red_border(self):
        source = np.zeros((240, 320, 3), np.uint8)
        state = OverlayState(fps=10., frame_age_ms=35., track_id=7, label='person',
            motion_state='MOVING', error=(20., -5., 20.6), target_velocity=(.2, 0.),
            pan=30, tilt=0, stable_seconds=4., stable_passed=False,
            alarm_directions=('RIGHT',), recording=True, capture_saved=False)
        output = draw_tracking_overlay(source, state, [(100, 100), (110, 100)])
        self.assertFalse(np.array_equal(output, source))
        self.assertTrue(np.all(source == 0))
        self.assertGreater(output[:4, :, 2].mean(), 200)

    def test_trajectory_is_bounded_and_none_clears(self):
        points = []
        for i in range(80):
            update_trajectory(points, (i, i), max_points=50)
        self.assertEqual(len(points), 50)
        update_trajectory(points, None)
        self.assertEqual(points, [])
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_overlay -v`

Expected: FAIL，提示 `tracking.overlay` 不存在。

- [ ] **Step 3: 实现叠加层**

```python
# tracking/overlay.py 公共状态
@dataclass(frozen=True)
class OverlayState:
    fps: float
    frame_age_ms: float
    track_id: int | None
    label: str | None
    motion_state: str
    error: tuple[float, float, float]
    target_velocity: tuple[float, float] | None
    pan: int
    tilt: int
    stable_seconds: float
    stable_passed: bool
    alarm_directions: tuple[str, ...]
    recording: bool
    capture_saved: bool
```

绘制函数先 `output = frame.copy()`；轨迹用青色折线，锁定框与预测点继续用紫色；报警时画 4 像素红框并显示 `BOUNDARY WARNING: RIGHT`。状态面板按规格使用绿/黄/红颜色，不绘制凭据或 RTSP 地址。

- [ ] **Step 4: 运行测试并提交**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_overlay -v`

Expected: 2 tests PASS。

```powershell
git add tracking/overlay.py test/tracking/test_overlay.py
git commit -m "feat: add tracking status overlay"
```

---

### Task 6: 主程序集成、实时抓拍、报警证据与报告

**Files:**
- Modify: `ball_camera_track.py`
- Modify: `test/test_ball_camera_track.py`
- Modify: `ptz/target_tracker.py`
- Modify: `ptz/test_target_tracker.py`

**Interfaces:**
- Consumes: Tasks 1–5 的全部公共接口。
- Produces CLI: `--no-record`、`--record-fps 10`、`--alarm-horizon .5`、`--alarm-margin .08`、`--alarm-cooldown 3`、`--low-light-threshold 55`。
- Produces artifacts: `annotated.mp4`、`events.csv`、`alarm-NNN.jpg`、实时 `tracking-error.csv/png`、单张 `center-capture.jpg`、增强后的 `report.json`。
- `PersistentTargetLock` 增加只读 `physical_key`；ID 重关联不变，真正释放后递增。
- Produces helpers: `safe_record(recorder, frame, failures) -> bool` 与 `close_tracking_resources(recorder, session, capture, stop_callable) -> list[str]`。

- [ ] **Step 1: 写 CLI、物理身份与离线零副作用测试**

```python
# test/test_ball_camera_track.py 新增
def test_acceptance_feature_defaults(self):
    args = build_parser().parse_args([])
    self.assertFalse(args.no_record)
    self.assertEqual(args.record_fps, 10.)
    self.assertEqual((args.alarm_horizon, args.alarm_margin, args.alarm_cooldown), (.5, .08, 3.))
    self.assertEqual(args.low_light_threshold, 55.)

def test_invalid_acceptance_parameters_are_rejected(self):
    for flags in (['--record-fps', '0'], ['--alarm-horizon', '0'],
                  ['--alarm-margin', '.5'], ['--alarm-cooldown', 'nan'],
                  ['--low-light-threshold', '256']):
        with self.assertRaises(ValueError):
            check_args(build_parser().parse_args(flags))
```

```python
# ptz/test_target_tracker.py 新增
def test_reassociation_preserves_physical_key_but_release_changes_it(self):
    lock = PersistentTargetLock(lost_grace=.2)
    first, changed = motion_target(7), motion_target(19)
    changed.observation.box = (14, 10, 54, 90)
    lock.update([first], first, 0.); key = lock.physical_key
    lock.update([changed], None, .1)
    self.assertEqual(lock.physical_key, key)
    lock.update([], None, .4)
    lock.update([motion_target(30)], motion_target(30), .5)
    self.assertNotEqual(lock.physical_key, key)
```

- [ ] **Step 2: 写资源清理和非致命录像失败测试**

```python
# test/test_ball_camera_track.py 新增导入 safe_record, close_tracking_resources
def test_optional_recording_failure_is_collected_not_raised(self):
    recorder = Mock()
    recorder.write.return_value = False
    failures = []
    frame = np.zeros((40, 60, 3), np.uint8)
    self.assertFalse(safe_record(recorder, frame, failures))
    self.assertEqual(failures, ['annotated.mp4'])
    recorder.write.assert_called_once_with(frame)

def test_cleanup_attempts_every_resource_when_recorder_close_fails(self):
    recorder, session, capture, stop = Mock(), Mock(), Mock(), Mock()
    recorder.close.side_effect = OSError('encoder close failed')
    failures = close_tracking_resources(recorder, session, capture, stop)
    recorder.close.assert_called_once_with()
    session.close.assert_called_once_with()
    capture.close.assert_called_once_with()
    self.assertEqual(stop.call_count, 3)
    self.assertEqual(failures, ['recorder close failed'])
```

- [ ] **Step 3: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.test_ball_camera_track ptz.test_target_tracker -v`

Expected: FAIL，缺少 CLI 参数、`physical_key` 和资源装配。

- [ ] **Step 4: 扩展持久锁物理身份**

```python
# ptz/target_tracker.py
class PersistentTargetLock:
    def __init__(self, lost_grace=2.0):
        self._physical_generation = 0
        self.physical_key = None

    def _acquire_new_physical_target(self, selected, timestamp):
        self._physical_generation += 1
        self.physical_key = f'target-{self._physical_generation}'
        self.track_id = selected.observation.track_id
        self.last_seen = self.acquired_at = timestamp
```

首次获取和释放后重新获取调用 `_acquire_new_physical_target`；`REASSOCIATED` 只更新 `track_id`，不得改变 `physical_key`。

- [ ] **Step 5: 集成参数、模块和逐帧数据流**

主循环顺序固定为：低照度增强 → detector → persistent lock → metrics → adaptive horizon → mailbox → alarm → capture → base annotate → overlay → recorder → imshow。报警事件立即 `session.append_event(...)` 并保存 `alarm-{index:03d}.jpg`；Windows 提示音用守护线程调用 `winsound.Beep(1200, 120)`，导入或播放失败只加入 `output_failures`。

抓拍条件保留：人物、中心死区、连续 3 帧、控制指令绝对值不超过 4、清晰度阈值。调用 `session.save_capture(raw)` 后不再更新候选，保证一次运行一张。

误差每个有效目标帧立即 `session.append_error(row)`；每 2 秒调用现有 `draw_error_curve`，退出时最终调用一次。UI 使用 `metrics.snapshot()`，录像写入最终叠加帧。

```python
def safe_record(recorder, frame, failures):
    try:
        written = recorder.write(frame)
    except (OSError, cv2.error, ValueError):
        written = False
    if not written and 'annotated.mp4' not in failures:
        failures.append('annotated.mp4')
    return written
```

- [ ] **Step 6: 将所有资源放入确定清理路径**

```python
def close_tracking_resources(recorder, session, capture, stop_callable):
    failures = []
    for name, action in (
            ('recorder close failed', None if recorder is None else recorder.close),
            ('session close failed', None if session is None else session.close),
            ('capture close failed', None if capture is None else capture.close)):
        if action is None:
            continue
        try:
            action()
        except (OSError, CameraError, cv2.error):
            failures.append(name)
    for _ in range(3):
        try:
            stop_callable()
        except CameraError:
            if 'PTZ stop failed' not in failures:
                failures.append('PTZ stop failed')
    return failures

# run() 的 finally 中按现有顺序先停止线程和看门狗，再调用：
output_failures.extend(close_tracking_resources(
    recorder, session, capture, client.stop))
```

每个清理动作各自包围 `try/except`，后一个资源不得因前一个关闭失败而跳过。报告新增 `average_fps`、`p95_frame_age_ms`、`p50/p95_radial_error_px`、`target_losses`、`reassociations`、`alarm_count`、`video_recording_created`、`video_frames`、`output_failures` 和 `low_light_frames`。

- [ ] **Step 7: 运行集成与全量测试**

Run: `.venv\Scripts\python.exe -m unittest test.test_ball_camera_track ptz.test_target_tracker test.tracking.test_alarm test.tracking.test_session test.tracking.test_metrics test.tracking.test_recorder test.tracking.test_adaptation test.tracking.test_overlay -v`

Expected: 全部 PASS。

同时明确核对现有行为测试仍通过：`TargetLockTests.test_other_salient_target_cannot_steal_active_lock`（多目标不得抢锁）、`TargetLockTests.test_lock_holds_through_short_gap_then_releases`（短时遮挡保持）、`PredictiveControllerTests.test_locked_static_target_enters_hysteretic_center_hold`（静止目标不等幅震荡）和 `TargetLockTests.test_nearby_same_person_box_reassociates_changed_tracker_id`（ID 更换后保持同一物理目标）。

Run: `.venv\Scripts\python.exe -m unittest discover -v`

Expected: 全部 PASS；若 `test_values_are_loaded_and_real_environment_wins` 因工作区真实 `.env` 污染环境而失败，先修复测试隔离（`patch.dict(os.environ, {}, clear=True)`）再重跑，不得忽略。

- [ ] **Step 8: 提交主程序集成**

```powershell
git add ball_camera_track.py test/test_ball_camera_track.py ptz/target_tracker.py ptz/test_target_tracker.py
git commit -m "feat: integrate tracking acceptance workflow"
```

---

### Task 7: 文档、实验矩阵与离线验收

**Files:**
- Modify: `README.md`
- Create: `docs/reports/tracking-acceptance-template.md`
- Create: `test/tracking/test_artifacts.py`

**Interfaces:**
- Documents: 实体连续跟踪命令、安全停止、UI 字段、报警规则、录像开关、输出文件与实验填写规则。
- Tests: 读取一次 12 帧合成会话，确认所有要求的产物存在且不含凭据。

- [ ] **Step 1: 写产物契约测试**

```python
# test/tracking/test_artifacts.py
import json, tempfile, unittest
from pathlib import Path

class ArtifactContractTests(unittest.TestCase):
    def test_required_artifact_names_and_report_fields(self):
        required = {'annotated.mp4', 'center-capture.jpg', 'events.csv',
                    'tracking-error.csv', 'tracking-error.png', 'report.json'}
        report_fields = {'stable_tracking_achieved', 'clear_capture_created',
                         'alarm_count', 'video_recording_created', 'average_fps',
                         'p95_frame_age_ms', 'p95_radial_error_px', 'output_failures'}
        self.assertEqual(required, set(REQUIRED_ARTIFACTS))
        self.assertTrue(report_fields.issubset(REQUIRED_REPORT_FIELDS))
```

将 `REQUIRED_ARTIFACTS` 和 `REQUIRED_REPORT_FIELDS` 定义在 `tracking/session.py`，由报告生成与测试共同使用，防止文档和实现漂移。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.venv\Scripts\python.exe -m unittest test.tracking.test_artifacts -v`

Expected: FAIL，常量尚未定义。

- [ ] **Step 3: 补齐产物常量与 README**

README 使用以下实际命令：

```powershell
.venv\Scripts\python.exe -u ball_camera_track.py --execute --continuous --show-all
.venv\Scripts\python.exe -u ball_camera_track.py --execute --continuous --show-all --no-record
```

说明 Q/Esc/关闭窗口/Ctrl+C，解释红色边框是未来 0.5 秒进入 8% 危险区，列出全部输出文件。删除 README 中旧的 0.2 秒预测、速度 24、20 秒上限和“非 MOVING 立即丢锁”等过时描述。

- [ ] **Step 4: 写实验报告模板**

模板包含固定表格列：场景、光照、人数、遮挡、运行命令、持续时间、平均 FPS、P95 帧龄、P50/P95 径向误差、最长连续跟踪、ID 重关联、报警方向、抓拍清晰度、录像可播放、停止后稳定、结论。固定场景为：匀速横移、突然变向、上下运动、静止、两人交叉、短时遮挡、正常光、低照度。

- [ ] **Step 5: 跑离线验收与性能对比**

Run: `.venv\Scripts\python.exe -m unittest discover -v`

Expected: 全部 PASS。

Run: `.venv\Scripts\python.exe -u ball_camera_track.py --output-dir output/acceptance-simulation`

Expected: 退出码 0，报告中 `camera_connections=0`、`ptz_move_requests_sent=0`，生成误差曲线且不生成真实报警或抓拍声明。

使用同一段测试视频分别运行 `--no-record --headless` 与默认录像模式，记录报告 `average_fps`；录像模式下降必须不超过 15%。如果超过，先把 `--record-fps` 调到 8 或把录像帧缩放到 480 宽，再重复比较，PTZ 控制频率和安全参数不得降低。

- [ ] **Step 6: 提交文档与验收契约**

```powershell
git add README.md docs/reports/tracking-acceptance-template.md tracking/session.py test/tracking/test_artifacts.py
git commit -m "docs: add tracking acceptance procedure"
```

---

### Task 8: 实体球机验收（需用户在场）

**Files:**
- Modify: `docs/reports/tracking-acceptance-template.md`（复制为带时间戳的实际报告，不覆盖模板）
- Inspect: `output/ball-track-<timestamp>/report.json`
- Inspect: `output/ball-track-<timestamp>/annotated.mp4`
- Inspect: `output/ball-track-<timestamp>/tracking-error.png`

**Interfaces:**
- Consumes: 已通过全量测试的实体跟踪命令。
- Produces: `docs/reports/tracking-acceptance-YYYY-MM-DD.md`，只记录实测结果，不把模拟数据写成实体结果。

- [ ] **Step 1: 做运行前安全检查**

确认只有一个跟踪进程、球机周围无障碍、网页端未同时控制云台、窗口 Q/Esc 可用。只读执行：

```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime,Path
.venv\Scripts\python.exe -u ball_camera_track.py
```

Expected: 第二条仅离线仿真，零相机连接和零 PTZ 请求。

- [ ] **Step 2: 运行实体连续验收**

```powershell
.venv\Scripts\python.exe -u ball_camera_track.py --execute --continuous --show-all
```

用户依次完成：匀速横移、突然变向、上下运动、停止、第二人经过、短时遮挡、走向边界、回到中心。窗口必须出现 `STABLE 10.0/10.0s PASS`、正确方向的边界报警和 `CAPTURE SAVED`。完成后按 Q 正常停止。

- [ ] **Step 3: 核对证据而非只看 PTZ 状态**

检查 `annotated.mp4` 可播放；`center-capture.jpg` 是未标注清晰人物图；`events.csv` 方向与录像一致；`tracking-error.png` 有 X/Y/径向曲线；`report.json` 的 `stable_tracking_achieved`、`clear_capture_created`、`video_recording_created` 均为 true，`failure` 为 null，停止后反馈稳定。相机实际移动仍以背景视觉位移证据为准，不以 HTTP 成功或 PTZ 数值变化替代。

- [ ] **Step 4: 填写并提交实际验收报告**

```powershell
git add docs/reports/tracking-acceptance-2026-09-16.md
git commit -m "docs: record physical tracking acceptance"
```

若某项失败，报告写入实际失败值和对应输出目录；回到产生该功能的任务添加失败测试再修复，不在报告中修改验收门槛。
