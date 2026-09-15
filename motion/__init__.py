"""统一运动检测：共享相机运动补偿 + 对象运动判定 + 显著目标锁定。

模块之间使用平铺导入（``import camera_motion`` 而不是 ``from motion import
camera_motion``）。这样从仓库根目录运行任一模块、以及被 ``test/motion``
下的测试导入时都能解析；直接以脚本方式运行时（``python motion/xxx.py``）
由该脚本内部的 ``sys.path`` 垫片补齐仓库根目录。
"""
