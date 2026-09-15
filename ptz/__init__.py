"""PTZ 云台控制与实机测试工具链（与目标检测相互独立）。

这些模块大量以脚本方式直接运行（``python ptz/ptz_smooth_test.py``），
因此每个入口脚本自带 ``sys.path`` 垫片把仓库根目录加入导入路径，
以便导入 ``hikvision_camera``、``ball_camera_detect`` 和 ``motion`` 包。
"""
