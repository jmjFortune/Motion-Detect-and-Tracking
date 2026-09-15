"""One background-only partial-affine estimate shared by tracking and motion.

Warps map previous-frame points to current-frame points in input-image pixels.
Failure returns identity to GMC consumers, but is explicitly unreliable and
has no valid residual evidence. No Ultralytics dependency is needed here.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


def _identity():
    return np.eye(2, 3, dtype=np.float32)


def _box_bounds(box, width, height):
    coordinates = np.asarray(box, dtype=float).reshape(-1)
    if coordinates.size != 4 or not np.isfinite(coordinates).all():
        return None
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        return None
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    return (left, top, right, bottom) if right > left and bottom > top else None


@dataclass
class CameraMotionResult:
    warp: np.ndarray
    reliable: bool
    reason: str
    tracked_points: int
    inlier_ratio: float
    coverage: float
    motion_mask: np.ndarray
    valid_mask: np.ndarray

    def transform_point(self, point):
        """Transform a previous measured center into current image coordinates."""
        transformed = self.warp[:, :2] @ np.asarray(point, dtype=np.float32) + self.warp[:, 2]
        return float(transformed[0]), float(transformed[1])

    def motion_ratio(self, box):
        """Fraction of valid pixels that changed within a full-frame xyxy box.

        Empty/invalid overlap yields zero; consumers must separately inspect
        reliability and valid overlap before interpreting this as stillness.
        """
        height, width = self.valid_mask.shape
        bounds = _box_bounds(box, width, height)
        if bounds is None:
            return 0.0
        left, top, right, bottom = bounds
        valid = self.valid_mask[top:bottom, left:right] != 0
        count = np.count_nonzero(valid)
        if not count:
            return 0.0
        motion = self.motion_mask[top:bottom, left:right] != 0
        return float(np.count_nonzero(motion & valid) / count)


class SharedCameraMotion:
    """BoT-SORT GMC adapter using masked, forward/backward-checked LK flow.

    Corner selection is tiled to retain dim background around textured objects.
    RANSAC requires >= min_points inliers, >= 65% agreement, >= 10% convex
    hull coverage and >= 30% span on both axes in both frames. Slow-camera
    plausibility limits are scale [0.85, 1.15], rotation <= 20 degrees and
    translation <= 25% of each dimension per frame. These are quality gates,
    not proof of correctness under strong parallax or repeating textures.
    """

    method = 'sparseOptFlow'

    def __init__(self, max_width=640, diff_threshold=25, min_points=12):
        if not np.isfinite(max_width) or max_width < 1 or int(max_width) != max_width:
            raise ValueError('max_width must be a positive integer')
        if not np.isfinite(diff_threshold) or not 0 <= diff_threshold <= 255:
            raise ValueError('diff_threshold must be finite and between 0 and 255')
        if not np.isfinite(min_points) or min_points < 3 or int(min_points) != min_points:
            raise ValueError('min_points must be an integer of at least 3')
        self.max_width = int(max_width)
        self.diff_threshold = float(diff_threshold)
        self.min_points = int(min_points)
        self.reset_params()

    def reset_params(self):
        """Clear frame history and per-stream call count, preserving settings."""
        self.calls = 0
        self.last_result = None
        self._previous_gray = None
        self._previous_background = None
        self._staged_detections = None

    def set_detections(self, boxes):
        """Stage all allowed YOLO xyxy boxes for the next GMC apply only.

        This overrides the high-confidence subset BoT-SORT passes to apply.
        An empty staged array intentionally overrides a nonempty subset.
        """
        self._staged_detections = np.asarray(boxes, dtype=float).copy()

    @staticmethod
    def _background_mask(shape, detections):
        height, width = shape
        mask = np.full(shape, 255, dtype=np.uint8)
        if detections is not None:
            boxes = np.asarray(detections, dtype=float)
            if boxes.size:
                if boxes.ndim == 1 and boxes.size == 4:
                    boxes = boxes.reshape(1, 4)
                if boxes.ndim != 2 or boxes.shape[1] != 4:
                    raise ValueError('detections must contain full-frame xyxy boxes')
                for box in boxes:
                    bounds = _box_bounds(box, width, height)
                    if bounds is not None:
                        left, top, right, bottom = bounds
                        mask[top:bottom, left:right] = 0
        # Keep LK support windows clear of semantic edges, not only centers.
        return cv2.erode(mask, np.ones((9, 9), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=255)

    @staticmethod
    def _corners(gray, mask):
        height, width = gray.shape
        mask = mask.copy()
        mask[:4] = mask[-4:] = 0
        mask[:, :4] = mask[:, -4:] = 0
        points = []
        for row in range(4):
            top, bottom = row * height // 4, (row + 1) * height // 4
            for column in range(4):
                left, right = column * width // 4, (column + 1) * width // 4
                if bottom <= top or right <= left:
                    continue
                corners = cv2.goodFeaturesToTrack(
                    gray[top:bottom, left:right], maxCorners=80, qualityLevel=0.01,
                    minDistance=7, mask=mask[top:bottom, left:right], blockSize=3)
                if corners is not None:
                    corners += np.float32([left, top])
                    points.append(corners)
        return np.concatenate(points) if points else None

    @staticmethod
    def _coverage(points, shape):
        height, width = shape
        coverage = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) /
                         (height * width))
        span = np.ptp(points, axis=0) / np.float32([width, height])
        return coverage, bool((span >= 0.3).all())

    @staticmethod
    def _track(previous, current, points, initial, previous_mask, current_mask):
        """Choose pyramid depth per point so LK support stays in background."""
        height, width = previous.shape
        source = points.reshape(-1, 2)
        destination = initial.reshape(-1, 2)
        finite = np.isfinite(destination).all(axis=1) & np.isfinite(source).all(axis=1)
        inside = (finite & (destination[:, 0] >= 0) & (destination[:, 0] < width) &
                  (destination[:, 1] >= 0) & (destination[:, 1] < height) &
                  (source[:, 0] >= 0) & (source[:, 0] < width) &
                  (source[:, 1] >= 0) & (source[:, 1] < height))
        levels = np.full(len(points), -1, dtype=int)
        previous_distance = cv2.distanceTransform(previous_mask, cv2.DIST_L2, 5)
        current_distance = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5)
        indices = np.flatnonzero(inside)
        src = np.floor(source[indices]).astype(int)
        dst = np.floor(destination[indices]).astype(int)
        distance = np.minimum(previous_distance[src[:, 1], src[:, 0]],
                              current_distance[dst[:, 1], dst[:, 0]])
        # 9px windows have radius four, plus pyramid filtering and refinement
        # margin. Avoiding only centers is insufficient at coarse levels.
        for level in range(4):
            levels[indices[distance >= 5 * (2 ** level) + 1]] = level
        following = initial.copy()
        status = np.zeros((len(points), 1), dtype=np.uint8)
        for level in range(4):
            subset = np.flatnonzero(levels == level)
            if not subset.size:
                continue
            tracked, valid, _ = cv2.calcOpticalFlowPyrLK(
                previous, current, points[subset], initial[subset].copy(),
                winSize=(9, 9), maxLevel=level, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if tracked is not None and valid is not None:
                following[subset], status[subset] = tracked, valid
        return following, status

    def _estimate(self, previous, current, previous_mask, current_mask):
        # LK assumes brightness constancy. Remove a global offset for tracking
        # too; the full-resolution residual uses its own aligned estimate.
        common = previous_mask & current_mask
        previous_values = previous[common != 0]
        current_values = current[common != 0]
        if not current_values.size:
            return None, 'no current background', 0, 0.0, 0.0
        offset = float(np.median(current_values)) - float(np.median(previous_values))
        tracking_current = np.clip(current.astype(np.float32) - offset, 0, 255).astype(np.uint8)
        # A masked, high-pass phase correlation seeds narrow-background LK
        # windows. It is never used as the output warp or quality evidence.
        common = previous_mask & current_mask
        seed_previous = previous.astype(np.float32)
        seed_current = current.astype(np.float32)
        seed_previous[common == 0] = np.median(previous[common != 0])
        seed_current[common == 0] = np.median(current[common != 0])
        seed_previous = (seed_previous - cv2.GaussianBlur(seed_previous, (0, 0), 3)) * (common != 0)
        seed_current = (seed_current - cv2.GaussianBlur(seed_current, (0, 0), 3)) * (common != 0)
        shift, response = cv2.phaseCorrelate(seed_previous, seed_current)
        height, width = current.shape
        if (not np.isfinite(shift).all() or not np.isfinite(response) or response < 0.1 or
                abs(shift[0]) > width * 0.25 or abs(shift[1]) > height * 0.25):
            shift = (0, 0)
        # Mask current objects in previous coordinates using the coarse seed;
        # using their unshifted boxes would unnecessarily erase newly visible
        # background tracks, especially in narrow strips beside a large object.
        inverse_seed = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        current_in_previous = cv2.warpAffine(current_mask, inverse_seed, (width, height),
                                            borderValue=255)
        previous_mask = previous_mask & np.where(current_in_previous == 255, 255, 0).astype(np.uint8)
        points = self._corners(previous, previous_mask)
        if points is None or len(points) < self.min_points:
            return None, 'insufficient background corners', 0 if points is None else len(points), 0.0, 0.0
        tracking_previous = previous.copy()
        tracking_previous[previous_mask == 0] = np.median(previous_values)
        tracking_current[current_mask == 0] = np.median(previous_values)
        initial = points + np.float32(shift)
        following, forward_status = self._track(tracking_previous, tracking_current, points,
                                               initial, previous_mask, current_mask)
        if following is None or forward_status is None:
            return None, 'forward optical flow failed', 0, 0.0, 0.0
        finite_forward = np.isfinite(following).all(axis=(1, 2))
        usable = (forward_status.ravel() != 0) & finite_forward
        points, following = points[usable], following[usable]
        if len(points) < self.min_points:
            return None, 'insufficient forward tracks', len(points), 0.0, 0.0
        returning, backward_status = self._track(tracking_current, tracking_previous, following,
                                                points, current_mask, previous_mask)
        if returning is None or backward_status is None:
            return None, 'backward optical flow failed', 0, 0.0, 0.0
        source, destination = points.reshape(-1, 2), following.reshape(-1, 2)
        height, width = current.shape
        consistent = ((backward_status.ravel() != 0) & np.isfinite(returning).all(axis=(1, 2)) &
                      (np.linalg.norm(returning.reshape(-1, 2) - source, axis=1) <= 1.5) &
                      (destination[:, 0] >= 0) & (destination[:, 0] < width) &
                      (destination[:, 1] >= 0) & (destination[:, 1] < height))
        indices = np.flatnonzero(consistent)
        pixels = np.floor(destination[indices]).astype(int)
        consistent[indices] &= current_mask[pixels[:, 1], pixels[:, 0]] != 0
        # Refinement must not move a window back onto a semantic edge.
        current_distance = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5)
        consistent[indices] &= current_distance[pixels[:, 1], pixels[:, 0]] >= 6
        source, destination = source[consistent], destination[consistent]
        count = len(source)
        if count < self.min_points:
            return None, 'insufficient consistent background tracks', count, 0.0, 0.0
        warp, inliers = cv2.estimateAffinePartial2D(
            source, destination, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=2000, confidence=0.99, refineIters=10)
        if warp is None or inliers is None or not np.isfinite(warp).all():
            return None, 'nonfinite or missing affine estimate', count, 0.0, 0.0
        inliers = inliers.ravel() != 0
        ratio = float(np.count_nonzero(inliers) / count)
        if np.count_nonzero(inliers) < self.min_points or ratio < 0.65:
            return None, 'insufficient RANSAC agreement', count, ratio, 0.0
        before_coverage, before_span = self._coverage(source[inliers], current.shape)
        after_coverage, after_span = self._coverage(destination[inliers], current.shape)
        coverage = min(before_coverage, after_coverage)
        if coverage < 0.1 or not before_span or not after_span:
            return None, 'insufficient spatial coverage', count, ratio, coverage
        scale = float(np.hypot(warp[0, 0], warp[1, 0]))
        angle = abs(math.degrees(math.atan2(warp[1, 0], warp[0, 0])))
        if (not 0.85 <= scale <= 1.15 or angle > 20 or
                abs(warp[0, 2]) > width * 0.25 or abs(warp[1, 2]) > height * 0.25):
            return None, 'implausible affine transform', count, ratio, coverage
        return warp, 'ok', count, ratio, coverage

    def _residual_masks(self, previous, current, previous_background, current_background, warp):
        height, width = current.shape
        aligned = cv2.warpAffine(previous, warp, (width, height))
        support = cv2.warpAffine(np.full(previous.shape, 255, np.uint8), warp, (width, height))
        valid = np.where(support == 255, 255, 0).astype(np.uint8)
        # Exclude interpolation support and the blurred residual's footprint.
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        aligned_background = cv2.warpAffine(previous_background, warp, (width, height))
        background = (valid != 0) & (current_background != 0) & (aligned_background == 255)
        if not background.any():
            return None, None
        difference = current.astype(np.float32) - aligned.astype(np.float32)
        offset = float(np.median(difference[background]))
        residual = cv2.GaussianBlur(np.abs(difference - offset), (5, 5), 0)
        moving = np.where((residual > self.diff_threshold) & (valid != 0), 255, 0).astype(np.uint8)
        return moving, valid

    def apply(self, frame, detections=None):
        """Estimate once, store full-size residual evidence, and return GMC warp."""
        if self._staged_detections is not None:
            detections = self._staged_detections
            self._staged_detections = None
        if frame.dtype != np.uint8 or frame.ndim not in (2, 3) or not frame.size:
            raise ValueError('frame must be a nonempty uint8 image')
        gray = frame.copy() if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = self._background_mask(gray.shape, detections)
        self.calls += 1
        zeros = np.zeros(gray.shape, np.uint8)
        result = CameraMotionResult(_identity(), False, 'first frame', 0, 0.0, 0.0,
                                    zeros.copy(), zeros)
        try:
            if self._previous_gray is None:
                return self._publish(result)
            if self._previous_gray.shape != gray.shape:
                result.reason = 'frame shape changed'
                return self._publish(result)
            height, width = gray.shape
            small_width = min(width, self.max_width)
            small_height = max(1, round(height * small_width / width))
            size = (small_width, small_height)
            previous_small = cv2.resize(self._previous_gray, size, interpolation=cv2.INTER_AREA)
            current_small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
            # Area sampling conservatively excludes any foreground contribution.
            previous_mask = cv2.resize(self._previous_background, size, interpolation=cv2.INTER_AREA)
            current_mask = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
            previous_mask = np.where(previous_mask == 255, 255, 0).astype(np.uint8)
            current_mask = np.where(current_mask == 255, 255, 0).astype(np.uint8)
            warp, reason, count, ratio, coverage = self._estimate(
                previous_small, current_small, previous_mask, current_mask)
            result.reason, result.tracked_points = reason, count
            result.inlier_ratio, result.coverage = ratio, coverage
            if warp is None:
                return self._publish(result)
            # Conjugate by actual x/y resize factors, including rounded height.
            scale = np.float64([small_width / width, small_height / height])
            full_warp = warp.copy()
            full_warp[:, :2] *= scale[np.newaxis, :] / scale[:, np.newaxis]
            full_warp[:, 2] /= scale
            moving, valid = self._residual_masks(self._previous_gray, gray,
                                                self._previous_background, background,
                                                full_warp.astype(np.float32))
            if valid is None:
                result.reason = 'no aligned background overlap'
                return self._publish(result)
            result.warp = full_warp.astype(np.float32)
            result.motion_mask, result.valid_mask = moving, valid
            result.reliable = True
            return self._publish(result)
        except cv2.error:
            result.reason = 'OpenCV motion estimation failed'
            return self._publish(result)
        finally:
            # Even an unreliable pair becomes the baseline for recovery.
            self._previous_gray = gray
            self._previous_background = background

    def _publish(self, result):
        self.last_result = result
        return result.warp
