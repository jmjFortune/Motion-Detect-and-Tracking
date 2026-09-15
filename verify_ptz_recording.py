"""Offline regression of PTZ video evidence. NEVER connects to a camera."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2

from ptz_motion_verification import VerificationFrame, create_foreground_detector, verify_sequence


def run(args):
    directory = args.output_dir or Path('output')/datetime.now().strftime('ptz-verify-%Y%m%d-%H%M%S')
    capture = cv2.VideoCapture(str(args.video))
    samples = []
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open local evidence video')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has invalid FPS')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            samples.append(VerificationFrame(len(samples)/fps, frame))
    finally:
        capture.release()
    if not samples:
        raise ValueError('Video has no decoded frames')
    detect = create_foreground_detector(args.model)
    def progress(count, total, verifier):
        print(f'OFFLINE verification {count}/{total}; good pairs='+str(verifier.summary()['reliable_pairs']), flush=True)
    report, rows = verify_sequence(samples, args.axis, detect, progress)
    report.update(mode='offline video replay', source=str(args.video), camera_requests_sent=0,
                  timestamp_source='encoded video FPS, not physical capture timing')
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (directory/'adjacent-motion.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    print('Results:', directory.resolve(), flush=True)
    return int(not report['reliable'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--axis', choices=('pan', 'tilt'), default='pan')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--output-dir', type=Path)
    raise SystemExit(run(parser.parse_args()))
