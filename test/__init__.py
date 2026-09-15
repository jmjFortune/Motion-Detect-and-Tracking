"""Test package for the unified-motion detector and the retained baseline.

Run from the project root (the folder that holds ``camera_motion.py``,
``object_motion.py`` and ``unified_motion_detector.py``)::

    uv run python -m unittest discover -s test -v

Modules here import the application modules from the project root, so they add
that root to ``sys.path`` themselves and also work when invoked directly::

    uv run python test/test_camera_motion.py
"""
