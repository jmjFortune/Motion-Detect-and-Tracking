# 课题 B：完整对象运动检测

## 静止 / 慢速移动相机统一检测（当前版本）

在 VSCode 的项目根目录终端中运行：

```powershell
uv run python unified_motion_detector.py --source 0
```

新版采用 YOLO-Nano + BoT-SORT + 共享背景 LK 光流 / RANSAC 部分仿射补偿。
不再按球机是否移动切换 MOG2。先跟踪完整的人、狗、猫，再区分相机运动与对象自身运动；站立挥手也通过框内变化量判定。选中后保持同一目标 ID，短暂丢失标为 LOST，不输出旧坐标冒充当前观测。

默认只显示运动对象：黄色为普通运动对象，红色为已锁定对象。`Q`、`Esc` 或关闭窗口退出。

调试时显示所有语义对象与补偿残差：

```powershell
uv run python unified_motion_detector.py --source 0 --show-all --show-mask
```

视频文件 / 无窗口 / 保存逐帧 CSV：

```powershell
uv run python unified_motion_detector.py --source input.mp4 --headless --max-frames 100 --csv output/unified.csv
```

海康球机子码流（请使用你自己的地址，不要公开账号密码）：

```powershell
uv run python unified_motion_detector.py --source "rtsp://用户名:密码@摄像机IP/Streaming/Channels/102"
```

### 指标与调参

- `speed`：相机补偿后的像素/秒，不是米/秒或云台角速度；`normalized_speed` 除以对象框对角线。
- `motion`：对象框内有效像素的补偿后变化比例，不是运动概率；局部残差块不是最终目标框。
- GMC=UNKNOWN：背景无纹理、快速移动或估计不可靠，不会据此新确认运动目标。短暂质量失败可保留旧状态并显示 LOW-QUALITY，持续失败进入 UNKNOWN。
- 默认 CPU、`yolo26n.pt`、`--imgsz 416`、处理宽度不超过 640；更小尺寸可用 `--imgsz 320 --resize-width 480`，代价是远处小目标更难检测。窗口 FPS 是本次实际处理速度。
- 检测其他类别：`--classes person,dog,cat,bird`，须是模型已有类别。
- 启动运动门槛：`--start-speed 0.08 --start-ratio 0.06 --start-frames 3`；停止门槛：`--stop-speed 0.04 --stop-ratio 0.025 --stop-frames 5`。速度与内部变化满足任一启动条件；停止要求两者都低。
- 视频元数据没有帧率时使用 `--source-fps 25`，填实际帧率。文件按源帧率计时，摄像头按实际采集间隔计时。

限制：补偿是相邻帧的 **部分仿射变换，不是完整单应性**，优先用于低速、固定变焦的左右 / 上下转动。大幅转动、强视差、模糊、局部照明和 YOLO 框抖动仍可能误判；不能等同真实球机验收。BoT-SORT 在这里用于身份关联，不是独立的运动分类器。

检测模块只输出目标 ID、框、运动状态和残余位移；**自身不发送 Pan/Tilt 指令**。独立双轴测试见下文；后续自动跟踪仍需方向、速度映射和视场角标定。

### 测试目录

统一运动检测的测试已集中到 `test/`：`test/test_camera_motion.py`（共享相机运动）、`test/test_object_motion.py`（对象运动与显著目标锁定）、`test/test_unified_motion_detector.py`（BoT-SORT 集成、真实 YOLO、CSV 与资源释放）。

`test/` 内的测试通过 `sys.path` 把项目根目录加入导入路径，因此可以从任意目录调用，也可以按包路径单独运行某个模块（`python -m unittest test.test_camera_motion`）。

根目录原有的 `test_camera_motion.py`、`test_object_motion.py`、`test_unified_motion_detector.py` 已删除（内容全部迁入 `test/`），不要再按可执行测试使用。

验证（不打开真实摄像头）：

```powershell
uv run python -m unittest discover -s test -v
uv run python -m unittest discover -s old -v
uv run python -m unittest discover -s test -p "test_camera_motion.py" -v
```

## 实体球机：电脑窗口显示检测框

已连接设备 `192.168.1.64`，型号 `DS-2DC2402IW-DE3`。程序在电脑端处理视频，**不会把检测框写回球机码流，所以网页预览仍是原始画面**。

```powershell
uv run python ball_camera_detect.py --show-all
```

凭据解析顺序：命令行参数 → 进程环境变量 `PTZ_PASSWORD` / `CAMERA_PASSWORD` / `CAMERA_USER` / `CAMERA_HOST` → 工作区 `.env` 文件 → 隐藏提示输入。`.env` 已在 `.gitignore` 中忽略，密码不写入报告、不打印、也不出现在错误信息里。默认子码流 102，关闭窗口或 Q/Esc 退出。此脚本只读视频和云台状态，绝不调用移动/停止接口。

```powershell
# 可选：把 CAMERA_HOST / CAMERA_USER / CAMERA_PASSWORD 写进工作区 .env，之后无需再输密码
uv run python ball_camera_detect.py --headless --show-all --duration 20
```

2026-09-15 第二次实机检查（统一运动检测流水线首次上真实球机）：207 帧/20.07 秒、10.31 FPS、GMC 可靠 192/207（92.8%，失败全部在启动前 2.8 秒）、10 个跟踪 ID、839 条人员观测、锁定 TRACKING 667 帧；运行前后云台反馈完全一致，移动请求 0。完整验收报告见 `output/ball-detect-20260915-210952/real-camera-acceptance.md`，其中记录了 9 帧残余速度异常（需人工看画面定性）以及尚未覆盖的转动/挥手/遮挡场景。

有限时长无窗口检查：

```powershell
uv run python ball_camera_detect.py --headless --show-all --duration 20
```

每次新建 `output/ball-detect-时间/`，含 CSV、标注截图和报告。2026-09-15 实机检查处理 191 帧/20.08 秒，约 9.51 FPS；识别到了完整人物运动框，检测前后云台原始反馈均为 `azimuth=2854,elevation=129,zoom=10`，发送移动请求数为 0。这是该次实测，不是性能或准确率保证。接流线程只保留最新帧，较慢的推理会丢弃积压帧而不是排队播放旧画面。

## 双轴二阶控制测试（独立于目标检测）

先模拟，不连接球机，也不需要密码：

```powershell
uv run python ptz_second_order_test.py --axis both --duration 12 --return-to-start
```

只读检查位置及设备边界，不移动：

```powershell
uv run python ptz_second_order_test.py --probe
```

**下列命令会真实转动球机**。确保周围无障碍，避免同时通过网页箭头或其他软件控制。默认只做低速、小角度测试；窗口 Q/Esc/X 或 Ctrl+C 会退出并尝试停止。

```powershell
uv run python ptz_second_order_test.py --execute --view --axis pan --pan-step 3 --duration 12 --return-to-start
uv run python ptz_second_order_test.py --execute --view --axis tilt --tilt-step 2 --duration 12 --return-to-start
```

`ptz_control.py` 使用临界阻尼二阶参考模型 `x'' + 2ωx' + ω²(x-r)=0`，位置和速度不因目标改变而重置；其后使用速度前馈 + PD 反馈，并限制软件速度和加速度命令。Pan/Tilt 各自维护状态，控制时钟独立于 YOLO 推理。测试输入是小角度阶跃，不是自动跟随检测目标。

注意：

- 软件默认名义速度上限 2°/s、加速度上限 1°/s²，最终 ISAPI 指令上限 6/100。实际转速映射 `--pan-gain/--tilt-gain` 尚未标定，不能把软件限幅宣称为电机真实物理限幅，也不保证物理 jerk。ISAPI 整数速度指令有量化，设备响应需看实测反馈。
- 默认状态单位为 0.1°，由 `--angle-unit` 调整。使用设备返回的坐标范围做预检，并在反馈偏移超过默认 6°、控制间隔超过 300ms 或反馈超时时中止。设备范围可能是通用协议范围，不等于精确机械边界；只在远离边界的小范围试验。
- 使用设备端 300ms `momentary` 脉冲；正常控制周期 100ms，脉冲重叠。断网/进程意外终止时脉冲按协议自行到期。`finally` 另发全零停止；不支持 momentary 时拒绝继续，不自动改成可能持续运行的 continuous 移动。协议依据：[海康 PTZ Service Specification 原始规范](https://www.scribd.com/document/834366266/Hikvision-Isapi-2-0-Ptz-Service)。
- `--return-to-start` 只是后半程把参考目标改回起点，不能保证有限时间内精确复位；报告记录最终偏差。紧急停止优先于平滑约束。
- 每次输出 `output/ptz-test-时间/trajectory.csv` 和 `report.json`；`--view` 显示实机码流，并保存 `physical-test.avi` 和截图。报告区分模拟/实机，含反馈角度、峰值命令和停止确认。停止请求被接受与机体真正停稳分别记录。

## 保留版本：MOG2 基线

旧代码原样保留在 `old/`，在项目根目录运行：

```powershell
uv run python old/motion_detector.py --source 0
```

以下是旧版说明；命令中的 `motion_detector.py` 应替换为 `old/motion_detector.py`。

该程序用于球机静止时自动发现运动目标。目前完成：

- MOG2 自适应背景建模；
- 阴影过滤和形态学去噪；
- 连通域运动区域提取；
- MOG2 与 YOLO 融合，只输出完整的人、狗、猫等语义目标框；
- 对象大小、运动量与位置连续性结合的显著目标选择；
- 摄像头、视频文件和 RTSP 视频流输入；
- 可选的标注视频与 CSV 输出。

球机开始转动后不要继续使用本模块进行背景减除。后续阶段应切换到目标跟踪、光流相机运动补偿、卡尔曼预测和云台控制。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本工作区已经创建了 `.venv`。如果 PowerShell 禁止执行激活脚本，也可以直接使用：

```powershell
.\.venv\Scripts\python.exe motion_detector.py --source 0
```

## 运行

电脑摄像头：

```powershell
python motion_detector.py --source 0
```

默认使用 `yolo26n.pt`，只显示 `person,dog,cat`，而不是皮肤、眼镜等零散运动区域。模型首次使用时会自动下载。

如果只想查看未经语义合并的原始运动块：

```powershell
python motion_detector.py --source 0 --mode raw
```

视频文件：

```powershell
python motion_detector.py --source input.mp4
```

海康 RTSP 子码流：

```powershell
python motion_detector.py --source "rtsp://用户名:密码@摄像机IP/Streaming/Channels/102"
```

保存结果：

```powershell
python motion_detector.py --source input.mp4 --output output/detected.mp4 --csv output/detections.csv
```

按 `Q` 或 `Esc` 退出。

## 常用调参

- 小目标检测不到：降低 `--min-area`，例如 `--min-area 400`。
- 完整对象没有被判为运动：降低 `--min-motion-ratio` 或 `--min-motion-pixels`。
- 需要检测其他类别：例如 `--classes person,dog,cat,bird`。
- 噪声误报较多：提高 `--min-area` 或 `--var-threshold`。
- 启动阶段误报：提高 `--warmup`，并保证背景建模时球机静止。
- 运行较慢：降低 `--resize-width`，例如 `--resize-width 640`。

程序默认预热 45 帧。预热期间不会输出目标框。

## 实体球机：网页接口短时运动测试

`ptz_web_test.py` 默认仅打印计划，不连接设备。显式执行左右、上下测试：

```powershell
uv run python ptz_web_test.py --execute --view --speed 30 --seconds 0.45
```

密码通过隐藏提示输入，不写入文件。使用球机网页的 `continuous` 接口；
四个方向分别运动，禁止同时转两轴、不变焦。独立进程按截止时间重复发送停止，
但这不是设备端自动到期保护，断网或进程被强制全部关闭仍可能导致停止失败。
每段通过视频背景位移验证，证据保存到 `output/ptz-web-时间/`。
仅水平测试加 `--axis pan`。速度限制 15–60，单段时间限制 0.2–0.6 秒。
运行前确保周围无障碍，不要同时用网页操纵球机。

2026-09-15 实测速度 30、每段 0.45 秒：左右和上下均有可靠的反向画面位移，
约 17–26 像素；结束后位置反馈稳定，未变焦。原二阶控制脚本保持不变，
其速度映射和位置方向尚未标定，不能据此宣称已实现自动目标跟踪。

## 连续速度＋二阶平滑＋S 曲线实体测试

```powershell
uv run python ptz_smooth_test.py --execute --view
```

先进行水平往返（约 9 秒），再进行垂直往返（约 8 秒）。默认每秒约 12 次更新
连续速度，S 曲线约束目标速度的变化，临界阻尼二阶模型进一步平滑目标速度。
正常结束时速度已渐降到零；按 Q/Esc、关闭窗口、视频过期、位置超限或控制停顿，
则紧急停止，不再坚持缓慢刹车。独立进程监测 0.65 秒心跳和硬截止时间并重复发停。
这些是软件停止保护，不能保证断网时停机。

不加 `--execute` 只生成离线 CSV，不连接球机；`--axis pan` 只测水平。
密码通过隐藏提示输入，结果保存在 `output/ptz-smooth-时间/`。
轨迹单位是 ISAPI 速度指令，不是已经标定的角速度。浮点指令的速度、加速度、
jerk 已做数值检查，但整数指令及实体运动的 jerk 不保证；此测试不包含精确位置闭环。

2026-09-15 实体测试完整运行：水平最大背景位移约 24 像素，垂直约 6 像素，
停止保护未触发，结束后反馈稳定且未变焦。平滑程度仍需结合现场观察判断；
水平有少量回位偏差，垂直低速响应较弱，下一步应分别标定两轴。

### 大幅位移验证修正

实体平滑测试不再仅比较起点与最大位移两张图。每轴运动期间保存新鲜的原始帧，
停止后用本地 YOLO 排除人、狗、猫区域和画面叠字，再对相邻帧做背景 LK/RANSAC，
按顺序复合变换。匹配失败或时间间隔超过 0.4 秒立即断开累计链，不跨缺失帧补位移，
不以云台位置反馈代替图像证据。需要至少 50% 的有效相邻帧对，以及至少 5 对、
持续 0.4 秒的可靠连续段，其轴向位移达到 3 像素。原 GMC 空间覆盖与一致性门槛不变。
报告中的累计位移是可靠连续段的最大位移，不是跨遮挡推算的整段角度。
检测和验证发生在停稳后，不阻塞运动控制；验证仍失败时不会执行下一轴。

只补测垂直方向、每方向匀速保持 2 秒（整轮往返约 12 秒，并非单方向 12 秒）：

```powershell
uv run python ptz_smooth_test.py --execute --view --axis tilt --tilt-speed 24 --cruise 2
```

可只回放已有录像验证，不连接或驱动球机：

```powershell
uv run python verify_ptz_recording.py --video output/ptz-smooth-20260915-203112/physical-test.avi --axis pan
```

该真实录像回归结果：205 帧、189 对可靠相邻帧（92.6%），通过连续背景位移验证。
