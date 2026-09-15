"""Real-time moving-object detection for a stationary camera.

This is the SEARCH stage of the PTZ-camera project.  It deliberately assumes
that the camera is not moving.  Once a target is locked and PTZ movement starts,
background subtraction must be paused and control handed to a tracker.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    area: int
    label: str = "motion"
    confidence: float = 1.0
    motion_ratio: float = 1.0

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def box_area(self) -> int:
        return self.width * self.height


@dataclass
class DetectorConfig:
    history: int = 500
    var_threshold: float = 25.0
    warmup_frames: int = 45
    min_area: int = 900
    max_area_ratio: float = 0.65
    open_size: int = 3
    close_size: int = 11
    dilate_iterations: int = 2


class MotionDetector:
    """MOG2 foreground segmentation plus connected-component filtering."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=True,
        )
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.open_size, config.open_size)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.close_size, config.close_size)
        )

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        self.frame_index += 1
        foreground = self.subtractor.apply(frame)

        # MOG2 marks shadows as 127 and foreground as 255.  A high threshold
        # removes shadows before morphology and connected-component extraction.
        _, mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.dilate(mask, self.open_kernel, iterations=self.config.dilate_iterations)

        if self.frame_index <= self.config.warmup_frames:
            return mask, []

        frame_area = frame.shape[0] * frame.shape[1]
        max_area = frame_area * self.config.max_area_ratio
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections: list[Detection] = []
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if area < self.config.min_area or area > max_area:
                continue
            if width < 8 or height < 8:
                continue
            detections.append(Detection(x, y, width, height, area))

        detections.sort(key=lambda item: item.area, reverse=True)
        return mask, detections


class SemanticMotionDetector:
    """Turn fragmented motion pixels into complete, meaningful object boxes.

    MOG2 answers "which pixels changed?" while YOLO answers "which object is
    this?".  An object is emitted only when enough foreground pixels overlap
    its YOLO bounding box.
    """

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        image_size: int = 416,
        confidence: float = 0.35,
        min_motion_pixels: int = 120,
        min_motion_ratio: float = 0.01,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Semantic mode requires Ultralytics. Install requirements.txt first."
            ) from exc

        self.model = YOLO(model_path)
        self.image_size = image_size
        self.confidence = confidence
        self.min_motion_pixels = min_motion_pixels
        self.min_motion_ratio = min_motion_ratio

        names = self.model.names
        name_to_id = {str(name).lower(): int(index) for index, name in names.items()}
        requested = [name.strip().lower() for name in class_names if name.strip()]
        missing = [name for name in requested if name not in name_to_id]
        if missing:
            raise ValueError(f"Classes not present in model: {', '.join(missing)}")
        self.class_ids = [name_to_id[name] for name in requested]

    def detect(self, frame: np.ndarray, motion_mask: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self.class_ids,
            device="cpu",
            verbose=False,
        )[0]

        frame_h, frame_w = frame.shape[:2]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().round().astype(int)
            x1 = int(np.clip(x1, 0, frame_w - 1))
            y1 = int(np.clip(y1, 0, frame_h - 1))
            x2 = int(np.clip(x2, x1 + 1, frame_w))
            y2 = int(np.clip(y2, y1 + 1, frame_h))

            object_mask = motion_mask[y1:y2, x1:x2]
            motion_pixels = int(cv2.countNonZero(object_mask))
            box_area = max((x2 - x1) * (y2 - y1), 1)
            motion_ratio = motion_pixels / box_area

            # Both conditions are required.  The ratio rejects a large object
            # touched by a tiny noise blob; the absolute count protects small
            # objects from passing on only a handful of changed pixels.
            if motion_pixels < self.min_motion_pixels:
                continue
            if motion_ratio < self.min_motion_ratio:
                continue

            class_id = int(box.cls.item())
            detections.append(
                Detection(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    area=motion_pixels,
                    label=str(self.model.names[class_id]),
                    confidence=float(box.conf.item()),
                    motion_ratio=motion_ratio,
                )
            )

        detections.sort(key=lambda item: item.box_area, reverse=True)
        return detections


class SalientTargetSelector:
    """Prefer a large, persistent target and avoid switching every frame."""

    def __init__(self, max_jump_ratio: float = 0.25, max_missing: int = 8) -> None:
        self.max_jump_ratio = max_jump_ratio
        self.max_missing = max_missing
        self.last_center: tuple[int, int] | None = None
        self.missing_frames = 0

    def select(
        self, detections: Iterable[Detection], frame_shape: tuple[int, ...]
    ) -> Detection | None:
        items = list(detections)
        if not items:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.last_center = None
            return None

        frame_h, frame_w = frame_shape[:2]
        diagonal = math.hypot(frame_w, frame_h)

        if self.last_center is None:
            selected = max(
                items,
                key=lambda item: item.box_area * (0.5 + min(item.motion_ratio, 0.5)),
            )
        else:
            def score(item: Detection) -> float:
                distance = math.dist(item.center, self.last_center) / diagonal
                continuity = max(0.0, 1.0 - distance / self.max_jump_ratio)
                area_score = item.box_area / (frame_w * frame_h)
                motion_score = min(item.motion_ratio, 0.25) / 0.25
                return 0.65 * continuity + 0.25 * area_score + 0.10 * motion_score

            selected = max(items, key=score)

            # If every candidate is implausibly far away, reacquire the largest.
            if math.dist(selected.center, self.last_center) / diagonal > self.max_jump_ratio:
                selected = max(items, key=lambda item: item.box_area)

        self.last_center = selected.center
        self.missing_frames = 0
        return selected


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, round(frame.shape[0] * scale)))


def annotate(
    frame: np.ndarray,
    detections: list[Detection],
    selected: Detection | None,
    fps: float,
    warming_up: bool,
) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        color = (0, 255, 255)
        thickness = 1
        if detection is selected:
            color = (0, 0, 255)
            thickness = 3
        cv2.rectangle(
            output,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
            thickness,
        )

    if selected is not None:
        center = selected.center
        cv2.circle(output, center, 5, (0, 0, 255), -1)
        target_text = (
            f"TARGET {selected.label} conf={selected.confidence:.2f} "
            f"motion={selected.motion_ratio * 100:.1f}%"
        )
        cv2.putText(
            output,
            target_text,
            (selected.x, max(24, selected.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    status = "WARMING UP" if warming_up else f"motion={len(detections)}"
    cv2.putText(
        output,
        f"{status}  FPS={fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect moving objects while the camera is stationary."
    )
    parser.add_argument("--source", default="0", help="Camera index, video path, or RTSP URL")
    parser.add_argument("--min-area", type=int, default=900, help="Minimum foreground area")
    parser.add_argument("--history", type=int, default=500, help="MOG2 history length")
    parser.add_argument("--var-threshold", type=float, default=25.0, help="MOG2 threshold")
    parser.add_argument("--warmup", type=int, default=45, help="Frames ignored during background warmup")
    parser.add_argument("--resize-width", type=int, default=960, help="Processing width; 0 keeps source size")
    parser.add_argument(
        "--mode",
        choices=("semantic", "raw"),
        default="semantic",
        help="semantic shows complete YOLO objects; raw shows foreground blobs",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model used in semantic mode")
    parser.add_argument(
        "--classes",
        default="person,dog,cat",
        help="Comma-separated YOLO class names accepted in semantic mode",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=416, help="YOLO inference size")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument(
        "--min-motion-ratio",
        type=float,
        default=0.01,
        help="Minimum moving-pixel fraction inside an object box",
    )
    parser.add_argument(
        "--min-motion-pixels",
        type=int,
        default=120,
        help="Minimum moving pixels inside an object box",
    )
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--csv", type=Path, help="Optional per-frame detection CSV")
    parser.add_argument("--headless", action="store_true", help="Run without display windows")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means unlimited")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DetectorConfig(
        history=args.history,
        var_threshold=args.var_threshold,
        warmup_frames=args.warmup,
        min_area=args.min_area,
    )

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    detector = MotionDetector(config)
    semantic_detector = None
    if args.mode == "semantic":
        semantic_detector = SemanticMotionDetector(
            model_path=args.model,
            class_names=args.classes.split(","),
            image_size=args.yolo_imgsz,
            confidence=args.yolo_conf,
            min_motion_pixels=args.min_motion_pixels,
            min_motion_ratio=args.min_motion_ratio,
        )
    selector = SalientTargetSelector()
    writer: cv2.VideoWriter | None = None
    csv_file = None
    csv_writer = None
    started = time.perf_counter()

    try:
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_file = args.csv.open("w", newline="", encoding="utf-8-sig")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["frame", "target_x", "target_y", "width", "height", "area"])

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            frame = resize_to_width(frame, args.resize_width)
            mask, raw_detections = detector.detect(frame)
            if semantic_detector is not None and detector.frame_index > config.warmup_frames:
                detections = semantic_detector.detect(frame, mask)
            elif semantic_detector is not None:
                detections = []
            else:
                detections = raw_detections
            selected = selector.select(detections, frame.shape)
            elapsed = max(time.perf_counter() - started, 1e-6)
            fps = detector.frame_index / elapsed
            warming_up = detector.frame_index <= config.warmup_frames
            display = annotate(frame, detections, selected, fps, warming_up)

            if csv_writer and selected:
                csv_writer.writerow([
                    detector.frame_index,
                    selected.x,
                    selected.y,
                    selected.width,
                    selected.height,
                    selected.area,
                ])

            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    if not source_fps or source_fps <= 1:
                        source_fps = 25.0
                    writer = cv2.VideoWriter(
                        str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (display.shape[1], display.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {args.output}")
                writer.write(display)

            if not args.headless:
                cv2.imshow("Motion detection", display)
                cv2.imshow("Foreground mask", mask)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.max_frames and detector.frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if not args.headless:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
