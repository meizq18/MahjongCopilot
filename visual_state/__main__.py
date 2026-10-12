import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from .core import Collector, WrapperClassifier, UnknownClassifier

def main():
    p = argparse.ArgumentParser(description='Read-only screenshots -> visual state JSONL')
    s = p.add_mutually_exclusive_group(required=True)
    s.add_argument('--frames', type=Path, help='Directory of ordered PNG files')
    s.add_argument('--live', action='store_true', help='Read display with mss; no input automation')
    p.add_argument('--rect', type=int, nargs=4, required=True, metavar=('X','Y','W','H'))
    p.add_argument('--wrapper', type=Path, help='Trusted majsoul_wrapper checkout; omit for segmentation-only ? labels')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval', type=float, default=.2)
    p.add_argument('--count', type=int, default=100)
    args = p.parse_args()
    if args.interval <= 0 or args.count <= 0:
        p.error('interval and count must be positive')
    collector = Collector(WrapperClassifier(args.wrapper) if args.wrapper else UnknownClassifier())
    def replay():
        files = sorted(args.frames.glob('*.png'))
        if not files:
            raise ValueError('No PNG frames found')
        for i, path in enumerate(files):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Unreadable frame: {path}')
            # Replay timestamps are synthetic. They cannot prove live continuity.
            yield frame, i*args.interval, str(path)
    def live():
        import mss
        with mss.mss() as screen:
            # rect coordinates are relative to the primary monitor image.
            for _ in range(args.count):
                yield np.asarray(screen.grab(screen.monitors[1]))[:, :, :3].copy(), time.monotonic(), None
                time.sleep(args.interval)
    with args.output.open('w', encoding='utf-8') as out:
        for frame, timestamp, path in replay() if args.frames else live():
            state = collector.observe(frame, tuple(args.rect), timestamp).to_dict()
            state['capture_source'] = 'replay' if path else 'live'
            state['image_path'] = path
            state['timestamp_source'] = 'synthetic' if path else 'monotonic'
            out.write(json.dumps(state, ensure_ascii=False)+'\n')
            out.flush()

if __name__ == '__main__':
    main()
