# 代码评审报告：统一运动检测中的 3 个缺陷

- 项目：课题 B「完整对象运动检测」（YOLO + BoT-SORT + 共享相机运动补偿）
- 仓库路径：`C:\Users\Administrator\Desktop\1`
- 被评审文件：`camera_motion.py`、`object_motion.py`、`unified_motion_detector.py`
- 评审日期：2026-09-15
- 评审方式：一个独立评审代理（只读、未修改任何文件）通读三个模块 + 已安装依赖源码（ultralytics 8.4.144、opencv 5.0.0.93）；随后由主代理**实际执行复现**，对评审结论做了核实与更正
- 本文档目的：把这三个缺陷交给外部评审（ChatGPT）判断修复方案，**特别是 F2 的修复与一条现有测试直接冲突，需要裁决**

---

## 0. 环境与可复现性

| 项目 | 值 |
| --- | --- |
| Python | 3.14（`.venv`） |
| 关键依赖 | numpy 2.5.3、opencv-python 5.0.0.93、ultralytics 8.4.144、lap 0.5.13 |
| 模型 | 本地 `yolo26n.pt`（5,544,453 字节，80 类 COCO） |
| 运行命令 | 在仓库根目录用 `.\.venv\Scripts\python.exe` 执行 |
| 当前测试基线 | `python -m unittest discover -s test` → **77 项全绿**；`python -m unittest discover -s old` → 3 项全绿 |
| 测试文件 | `test/test_camera_motion.py`(20)、`test/test_object_motion.py`(34)、`test/test_unified_motion_detector.py`(15)、`test/test_hikvision_camera.py`(8)；基线 `old/test_motion_detector.py`(3) |

> 说明：评审代理所在环境无法启动任何子进程（`STATUS_DLL_INIT_FAILED`），所以它的结论全部来自源码/依赖轮子的静态阅读；下文中标注「**实测**」的结论都是主代理在本机真实跑出来的。

---

## 1. 三个缺陷速览

| 编号 | 级别 | 一句话 | 是否已实测复现 | 修复量 |
| --- | --- | --- | --- | --- |
| F1 | 低（原评审定为关键，已下调） | RTSP 凭据抑制调用了不存在的 OpenCV API，是死代码；评审怀疑会泄漏凭据，**我未能复现泄漏** | 缺陷本身已证实；泄漏未复现 | 2~6 行 |
| F2 | **中高** | 历史按 `last_seen` 清理时**不排除当前帧可见的目标**，帧间隔 > 2 秒时功能静默失效 | **已复现，100% 失效** | 1 行 + 需裁决测试冲突 |
| F3 | 低 | 不带 `://` 的凭据串会被应用自己的报错原样打印 | **已复现** | 2 行 |

---

## 2. F1：OpenCV 日志抑制调用了不存在的 API（死代码）

### 2.1 位置

`unified_motion_detector.py:126-144`，关键两行：

```python
def open_capture(source):
    """Keep native OpenCV/FFmpeg messages from printing URL credentials."""
    if isinstance(source, str) and '://' in source:
        # Set before opening the backend. OpenCV's own logs have a separate level.
        level = getattr(cv2, 'getLogLevel', lambda: None)()          # 第 130 行
        previous = os.environ.get('OPENCV_FFMPEG_LOGLEVEL')
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
        getattr(cv2, 'setLogLevel', lambda value: None)(0)           # 第 133 行
        try:
            return cv2.VideoCapture(source)
        except cv2.error:
            raise RuntimeError('Cannot open network video stream') from None
        finally:
            if level is not None:
                getattr(cv2, 'setLogLevel', lambda value: None)(level)  # 第 140 行
            if previous is None:
                os.environ.pop('OPENCV_FFMPEG_LOGLEVEL', None)
            else:
                os.environ['OPENCV_FFMPEG_LOGLEVEL'] = previous
```

### 2.2 问题

`cv2.setLogLevel` / `cv2.getLogLevel` 在 **opencv-python 5.0.0.93 的顶层命名空间里不存在**，真正的 API 在 `cv2.utils.logging` 下。因为代码用了 `getattr(cv2, 'setLogLevel', lambda value: None)` 这种带兜底的写法，**缺 API 不会报错，而是静默走空实现**，所以这两行从来没生效过，也没人发现。

**实测（本机）**：

```
root setLogLevel: False
cv2.utils.logging setLogLevel: True
cv2.utils.logging.LOG_LEVEL_SILENT = 0
```

只有 `OPENCV_FFMPEG_LOGLEVEL=-8` 是真实生效的，而且它在 `cv2.VideoCapture()` 返回后立即被还原（第 138-144 行的 `finally`），所以**视频流建立之后的运行期日志完全不受抑制**——中途 RTSP 断开、FFmpeg 报错时，日志里可能带着完整源串（含 `user:password`）。

### 2.3 影响（更正原评审的定级）

- 原评审把它定为 **CRITICAL「已泄漏凭据」**。
- **实测结果：我没有复现出泄漏。** 用 `rtsp://admin:SUPERSECRET123@127.0.0.1:1/none` 分别走应用路径 `open_capture(...)` 和裸 `cv2.VideoCapture(...)`，把 OS 级 stderr 重定向到文件后检查：**两种路径的 stderr 里都没有这个密钥**，只看到 `[ WARN:0@30.058] global cap_ffmpeg_impl.hpp:453 _opencv_ffmpeg_interrupt_callback Stream timeout triggered after 30051.307000 ms`。
- 原因推测：该 URL 在这次运行中没有走「会打印文件名的 FFmpeg 打开路径」（原始 `cv2.VideoCapture` 同样没打印），所以打印文件名的那条 WARN 根本没发生。
- 因此定级下调为**低**：目前是「意图失效的死代码 + 未证实的风险」，而不是已发生的安全事件。风险窗口存在，但需要「FFmpeg 后端被启用」或「流中途失败」这类条件。

### 2.4 建议修复

```python
import cv2.utils.logging as cv2_logging  # 真实 API 位置

def open_capture(source):
    if isinstance(source, str) and '://' in source:
        previous_level = cv2_logging.getLogLevel()
        cv2_logging.setLogLevel(cv2_logging.LOG_LEVEL_SILENT)
        ...
```

要点：
1. 用 `cv2.utils.logging.setLogLevel/getLogLevel`，不要再用 `getattr` 兜底（兜底会把「API 名字写错」变成静默失效——这次就是这么漏掉的）。
2. 抑制范围应覆盖**整个网络流生命周期**，而不只是 `open` 那一瞬；否则运行期中途断流的日志仍不受保护。
3. 恢复时注意与 `OPENCV_FFMPEG_LOGLEVEL` 的环境变量还原保持对称。

### 2.5 覆盖该缺陷的回归测试（修前必然失败）

```python
def test_network_open_silences_opencv_logger(self):
    recorded = []
    with patch('unified_motion_detector.cv2.utils.logging.setLogLevel', recorded.append), \
         patch('unified_motion_detector.cv2.VideoCapture'):
        open_capture('rtsp://user:secret@127.0.0.1:1/x')
    self.assertIn(cv2.utils.logging.LOG_LEVEL_SILENT, recorded)   # 今天是空列表
```

更强的端到端版本（`contextlib.redirect_stderr` 看不到原生日志，必须捕获 OS 级 fd 2）：

```python
def test_network_open_never_prints_the_secret(self):
    result = subprocess.run([sys.executable, '-c',
        "from unified_motion_detector import open_capture;"
        "open_capture('rtsp://admin:S3cret@127.0.0.1:1/none')"],
        capture_output=True, text=True, cwd=str(APP_ROOT), timeout=120)
    self.assertNotIn('S3cret', result.stderr)
```

### 2.6 请 ChatGPT 判断的问题

1. 只做「用正确 API + 保持抑制整个流生命周期」是否足够，还是应该改为完全不把凭据放进 OpenCV（例如用 `CAP_PROP_*` 之外的机制、或先连接再注入账号）？
2. 上面这个端到端测试在 CI 上依赖 30 秒级超时（不可达地址要等超时），有没有更快的确定性做法？

---

## 3. F2：历史清理会删掉「当前帧正在被观测」的目标（已复现，功能静默失效）

### 3.1 位置

`object_motion.py:134-141`（清理循环）与 `object_motion.py:191`（刷新 `last_seen`）：

```python
observations = list(observations)
visible = {observation.track_id for observation in observations}
for track_id, history in list(self.histories.items()):
    if timestamp-history.last_seen > self.config.history_ttl:   # 第 135 行
        del self.histories[track_id]                            # 第 136 行
    elif track_id not in visible:
        history.previous_center = history.previous_time = None
        history.previous_reliable = False
        history.break_evidence()
        self._expire_quality(history, timestamp)
```

`history_ttl` 默认 **2.0 秒**（`MotionConfig`）。

### 3.2 问题

删除条件只判断「距上次见到超过 TTL」，**没有判断这个 ID 在本次调用里是否可见**，而且判断发生在 `last_seen` 刷新（第 191 行）之前。于是当一个 ID 的相邻两次观测间隔超过 2 秒时：

1. 它的历史被删掉；
2. 紧接着第 145 行 `self.histories.setdefault(observation.track_id, _History(timestamp))` 用当前时间戳重建了一个全新历史；
3. 新历史的 `previous_center/previous_time` 为空 → `dt = 0` → `reliable = False` → 状态只能停在 `UNKNOWN`；
4. 下一次观测重复同样的过程，**永远无法累积到 `start_frames` 次确认**，因此永远不会进入 `MOVING`。

后果链条：没有 MOVING → `SalientTargetSelector` 永远拿不到候选（它只接受可靠 MOVING）→ **不锁定任何目标 → 界面上一个框都不画 → CSV 全部是 UNKNOWN**，而且**不报任何错、不打任何日志**。

### 3.3 实测复现（本机真实执行）

同一物体每帧移动 30px（归一化速度远高于启动门槛 0.08，确保不是调参问题）：

| 帧间隔 | 状态序列（6 帧） | 结论 |
| --- | --- | --- |
| 0.1 s | UNKNOWN, UNKNOWN, UNKNOWN, **MOVING**, MOVING, MOVING | 正常 |
| 1.0 s | UNKNOWN, UNKNOWN, UNKNOWN, **MOVING**, MOVING, MOVING | 正常 |
| 2.5 s | UNKNOWN ×6 | **坏** |
| 3.0 s | UNKNOWN ×6 | **坏** |
| 2.5 s，`history_ttl=60` | UNKNOWN, UNKNOWN, UNKNOWN, **MOVING**, MOVING, MOVING | 证明就是这条 TTL 删除逻辑导致 |

对照实验（把 TTL 调到 60 秒后同一输入恢复正常）把原因锁定在清理逻辑，而不是速度门槛或补偿。

复现脚本：

```python
import numpy as np
from camera_motion import CameraMotionResult
from object_motion import ObjectMotionClassifier, ObjectObservation, MotionConfig

def cam():
    shape = (240, 320)
    return CameraMotionResult(np.float32([[1, 0, 0], [0, 1, 0]]), True, 'ok',
                              30, .95, .5, np.zeros(shape, np.uint8),
                              np.full(shape, 255, np.uint8))

def states(gap, ttl=None):
    config = None if ttl is None else MotionConfig(history_ttl=ttl)
    c = ObjectMotionClassifier(config)
    out = []
    for i in range(6):
        obs = ObjectObservation(1, (20 + 30 * i, 30, 60 + 30 * i, 90), 'person', .9)
        out.append(c.update([obs], cam(), i * gap)[0].state)
    return out

print(states(0.1))          # ['UNKNOWN','UNKNOWN','UNKNOWN','MOVING','MOVING','MOVING']
print(states(2.5))          # ['UNKNOWN'] * 6        ← 缺陷
print(states(2.5, ttl=60.)) # ['UNKNOWN','UNKNOWN','UNKNOWN','MOVING','MOVING','MOVING']
```

### 3.4 触发条件（**比原评审描述的要窄，已更正**）

`timestamp` 的来源在两类源上不同（`unified_motion_detector.py:148-161` 的 `FrameClock`）：

- **实时源**（摄像头 / RTSP）：用 `time.monotonic()` 的采集时间差，所以帧间隔反映**真实处理速度**。要超过 2 秒需要管线慢到 **< 0.5 FPS**（实测上次实机运行为 10.31 FPS，差约 20 倍）。
- **文件源**：用 `帧号 ÷ 源帧率`，与处理速度无关，只取决于视频元数据的 FPS。所以**只要视频 FPS < 0.5（延时摄影、或 `--source-fps` 填得过小）就必然触发**。

结论：**发生概率低，但一旦发生就是 100% 的功能失效且无任何提示**。原评审推测的「CPU 慢 + `--imgsz 1280`」在实时源上要慢到 0.5 FPS 以下才成立，比它说的更极端。

### 3.5 建议修复 ⚠️ 与现有测试冲突（**需要 ChatGPT 裁决**）

最直接的修法：

```python
for track_id, history in list(self.histories.items()):
    if track_id in visible:                     # 本帧可见：绝不清理
        continue
    if timestamp - history.last_seen > self.config.history_ttl:
        del self.histories[track_id]
        continue
    history.previous_center = history.previous_time = None
    history.previous_reliable = False
    history.break_evidence()
    self._expire_quality(history, timestamp)
```

**但这个修法会让一条现有测试失败**，而那条测试看起来是作者有意写下的行为约定（`test/test_object_motion.py::test_single_confirmation_and_zero_grace_are_supported`）：

```python
classifier = ObjectMotionClassifier(MotionConfig(start_frames=1, stop_frames=1,
                                                unreliable_grace=0., history_ttl=0.))
classifier.update([observation()], self.camera, 0.)
target = classifier.update([observation(25)], self.camera, .1)[0]
# Zero TTL expires even visible history if last_seen is in the past.
self.assertEqual(target.state, 'UNKNOWN')
```

即：当 `history_ttl=0` 时，**连本帧可见的目标也要被清理**（作者用注释明确写了这是有意行为）。所以这里有真正的语义冲突，必须先决定哪一种是正确语义：

- **方案 A**：可见目标永不清理（修掉 F2）。代价：`history_ttl=0` 这条测试必须改，等于放宽作者写下的约定；同时「TTL 只用于清理过期 ID」的语义更一致，但**还需要回答**：实时源长时间丢帧（例如断流 3 秒后同一 ID 回来）时，是否应该认为证据连续？
- **方案 B**：保持现状，认为「超过 TTL 就是证据断裂」。代价：F2 不修，但必须**至少**把失效变成可见——例如 `RUN` 层检测到「帧间隔 > `history_ttl`」时打印明确警告，或直接拒绝这种配置组合（启动时校验 `source_fps` 与 `history_ttl` 的关系）。
- **方案 C**：把语义拆成两个参数——`evidence_gap`（证据连续性上限，仅用于判断是否断裂）与 `history_ttl`（仅用于内存回收，取 `max(evidence_gap, ...)`）。语义最清楚，但要改配置面和 CLI。

**请 ChatGPT 明确：这三种方案各自的风险，以及应该选哪个。** 这是我目前最需要外部意见的地方，因为修复动作会改动一条既有测试所固定的行为，而任务约束里明确禁止「为了让测试通过而削弱测试」。

### 3.6 覆盖该缺陷的回归测试（方案 A 下，修前必然失败）

```python
def test_visible_track_history_survives_gap_longer_than_ttl(self):
    classifier = ObjectMotionClassifier()
    states = [classifier.update([observation(20 + 15 * i)], self.camera, i * 1.0)[0].state
              for i in range(6)]
    self.assertEqual(states[-1], 'MOVING')   # 今天得到 UNKNOWN
```

---

## 4. F3：不带 `://` 的凭据串会被原样打印（已复现）

### 4.1 位置

`unified_motion_detector.py:117-123`（`safe_source_label`）与 `unified_motion_detector.py:326`（`main` 里的遮蔽条件），错误信息在第 263 行拼装：

```python
def safe_source_label(source):
    if isinstance(source, int):
        return f'camera index {source}'
    parsed = urlsplit(source)
    if parsed.scheme and '://' in source:      # 只有带 scheme 才遮蔽
        return 'network video stream'
    return str(source)                          # 否则原样返回
```

```python
message = ('Network video processing failed; check connection and settings.'
           if '://' in args.source else str(error))   # 第 326 行：只看 '://'
```

### 4.2 问题与实测

遮蔽逻辑完全依赖字符串里是否存在 `://`。如果用户粘贴地址时**把 `rtsp://` 掉了**（真实的复制粘贴失误），凭据就会被原样打印。

**实测（本机真实执行）**：

```
'rtsp://admin:secret123@192.168.1.64/Streaming/Channels/102' -> 'network video stream'          ✅
'admin:secret123@192.168.1.64/Streaming/Channels/102'        -> 'admin:secret123@192.168.1.64/Streaming/Channels/102'  ❌ 原样
'C:/videos/clip.mp4'                                        -> 'C:/videos/clip.mp4'           （正常，不含凭据）
'0'                                                         -> '0'                             （正常）
```

打印出来的是 `Error: Cannot open video source: admin:secret123@192.168.1.64/...`，密码明文进入终端回显、会被复制进聊天记录/工单/截图。

### 4.3 影响

低危：要求用户输入形态异常（缺 scheme）才触发；但一旦触发就是明文凭据外泄，且日志通常会被转发给他人。局域网实验环境下风险有限。

### 4.4 建议修复

思路是「不依赖 scheme 判断，而是**任何源串在回显前都剥掉 `user:pass@`**」：

```python
def safe_source_label(source):
    if isinstance(source, int):
        return f'camera index {source}'
    text = str(source)
    if '://' in text or '@' in text:
        return 'network video stream'
    return text
```

要点：
1. 判断条件从「有没有 `://`」改成「**有没有 userinfo（`@`）或 scheme**」；
2. 第 326 行的错误遮蔽条件要同步改成同一个判断函数，避免两处规则不一致（现在就是不一致导致的漏洞）；
3. 建议把「是否敏感源」抽成一个函数，`safe_source_label` 和 `main` 共用，避免以后再次分叉。

### 4.5 覆盖该缺陷的回归测试（修前必然失败）

```python
def test_credential_bearing_source_without_scheme_is_not_echoed(self):
    label = safe_source_label('admin:secret@192.168.1.64/Streaming/Channels/101')
    self.assertNotIn('secret', label)
    self.assertNotIn('admin', label)
```

---

## 5. 原评审列出但我判定「暂不改」的次级问题

这些是加固项，没有找到真实触发路径，记录在此供参考，不作为本次修复目标：

| 位置 | 现象 | 为什么暂不改 |
| --- | --- | --- |
| `unified_motion_detector.py:86-87,107` | BoT-SORT 的 GMC 钩子会吞掉 `gmc.apply` 的**任意**异常并回退为单位变换（已安装的 `byte_tracker.py:358-363` 是 `try/except Exception`），此时 `self.camera.last_result` 仍停留在上一帧，`process()` 会把这个**过期但可能 `reliable=True`** 的结果喂给分类器 | `apply()` 在所有现实路径上都会发布结果，且内部已捕获 `cv2.error`，构造不出可达触发 |
| `camera_motion.py:303-307` | 暂存的完整语义框在**帧合法性校验之前**被消费，一帧被拒会静默丢掉暂存集合 | CLI 路径不可达 |
| `camera_motion.py:82-84` | 非数值构造参数下 `np.isfinite` 抛 `TypeError` 而非 `ValueError` | CLI 不可达 |
| CSV 字段缺 `center` | 设计文档要求输出选中目标的中心，`CSV_FIELDS` 没有中心列（可由 `x1..y2` 推出） | 计划文档的 Task-3 字段清单里本就没有中心，属「设计 vs 计划」的表述差异，非阻塞 |

---

## 6. 原评审确认无误的部分（避免 ChatGPT 误改）

以下是静态+实测核对通过的点，**不要**建议改动：

- **每帧只估计一次相机运动且两个消费者共用**：`gmc` 只在 `bot_sort.py:169/212` 与 `byte_tracker.py:358-365` 出现；钩子无条件调用（零检测时也调用）；`UnifiedMotionDetector.process` 先更新跟踪器再读取**同一个** `self.camera.last_result` 给分类器。
- **是部分仿射，不是完整单应**：`estimateAffinePartial2D(..., RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99, refineIters=10)` + 缩放/旋转/平移合理性门槛；全仓库无 `findHomography`。设计文档明确「完整单应性留作后续对比实验」，所以**不要建议改成单应矩阵**。
- **补偿失败 → 单位变换且不累积证据**：失败路径都返回 `reliable=False` + 单位阵 + 新建零掩膜；`object_motion.py:150` 要求 `camera.reliable` 才计算残差；不可靠分支调用 `break_evidence()`，运动/静止计数器都不累加。
- **检测行 → 跟踪输出的索引映射正确**：`[x1,y1,x2,y2,track_id,score,cls,idx]`，`idx` 是全检测集内的下标（`byte_tracker.py:277-279`），`row[7]` 用法正确；8 列断言正确（`Boxes` 没有 `xywhr`）。
- **速度与时间基准**：用**实测检测框**（不是卡尔曼平滑值），要求 `dt > 0`，目标缺席即清空 `previous_center/time`；文件用 `帧号/FPS`，实时用单调时钟。
- **LOST 不带过期坐标**：`object_motion.py:226` 返回 `target=None`，CSV 由 `DictWriter` 的 `restval` 写成空值——**已用真实相机数据核对**：52 条无目标 LOST 行的 `track_id/x1/y1/x2/y2` 全为空。
- **资源释放**：`run()` 单个 `try/finally` 覆盖 `capture.release()`、CSV 关闭、窗口销毁；`try` 内无提前 `return`。
- **退出键**：Q / Esc / 窗口关闭按钮均已处理。注意计划里写的「Q、Esc、X」中的 **X 指的是窗口右上角关闭按钮**，全仓库没有任何脚本绑定 `x` 键（grep `ord('x')` 命中 0 次）——这不是缺陷。
- **CLI 全参数生效**、正数/有限值/非负宽限校验齐备；无自动打开摄像头、无 PTZ 指令、无新增模型下载。

---

## 7. 需要 ChatGPT 回答的问题清单

1. **F2 的语义裁决（最重要）**：第 3.5 节的方案 A / B / C 选哪个？「可见目标永不清理」与现有 `history_ttl=0` 测试的约定哪个才是正确的？如果选 A，那条测试应该怎么改才不算「削弱测试」？
2. **F2 的实时源语义**：实时流丢帧 3 秒后同一 ID 回来，应该视为「证据连续」还是「断裂」？这决定了 `history_ttl` 到底该以「证据连续性」还是「内存回收」为语义。
3. **F2 的额外防线**：除了修清理逻辑，是否应该在启动时校验「源 FPS / 预期处理速度」与 `history_ttl` 的关系，或者在这种配置下直接给出显式警告？
4. **F1 的完整修复面**：只改 API 调用是否足够？有没有办法从根上不让凭据进入 OpenCV/FFmpeg 的日志路径？以及 2.6 节那个端到端测试如何做才能在 CI 上又快又确定？
5. **F3 的自检方式**：把「是否有任何源串在回显前未剥离 userinfo」变成一条**结构性**约束（例如所有打印路径必须经过同一个函数）是否更稳妥？有没有推荐的做法？
6. **是否存在我们都没看到的第四个问题**：特别是「相机运动与对象运动耦合」的部分（背景特征会挖掉所有被跟踪目标的框，目标框铺满画面会导致 GMC 覆盖不足）。

---

## 8. 附：真实相机实测数据（供判断严重程度参考）

2026-09-15 用实体球机（DS-2DC2402IW-DE3，子码流 102）跑 `ball_camera_detect.py --headless --show-all --duration 20`：

| 指标 | 值 |
| --- | --- |
| 处理 | 207 帧 / 20.07 s = 10.31 FPS（CPU、yolo26n、imgsz 416） |
| 相机运动补偿可靠率 | 192/207 = 92.8%，15 次失败全部在启动前 2.78 s 内 |
| 跟踪 ID | 10 个；目标观测 839 条（全是 person） |
| 状态 | STATIC 452 / MOVING 214 / UNKNOWN 173 |
| 锁定 | TRACKING 667 / LOST 65 / NONE 120 |
| 云台指令 | 0（前后位置反馈完全一致） |
| 未定性项 | 9 帧（1.07%）残余归一化速度 > 0.4，且这些帧 GMC 可靠、框内变化极低，需人工看画面定性 |

因为本次运行实测 10.31 FPS（远高于 F2 的 0.5 FPS 门槛），所以 **F2 在本次实机运行中没有触发**；实机数据里也没有观察到凭据泄漏。
