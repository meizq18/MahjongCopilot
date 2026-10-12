from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import Protocol
import importlib.util
from pathlib import Path
import math
import numpy as np
import cv2

VALID = {f'{n}{s}' for s in 'mps' for n in range(1, 10)} | {f'{n}z' for n in range(1, 8)} | {'0m', '0p', '0s'}

def mjai_tile(tile: str) -> str:
    if tile not in VALID:
        raise ValueError(f'Unknown tile: {tile}')
    if tile.endswith('z'):
        return ['E', 'S', 'W', 'N', 'P', 'F', 'C'][int(tile[0])-1]
    return '5' + tile[1] + 'r' if tile[0] == '0' else tile

@dataclass
class Observed:
    value: object = None
    source: str = 'unobserved'
    frame_id: int | None = None
    confidence: float | None = None  # None: classifier has no calibrated confidence

@dataclass
class Player:
    # Screen-relative index, NOT MJAI actor; absolute seat mapping is unknown.
    screen_position: str
    score: Observed = field(default_factory=Observed)
    river: Observed = field(default_factory=Observed)
    melds: Observed = field(default_factory=Observed)
    reached: Observed = field(default_factory=Observed)

@dataclass
class State:
    frame_id: int
    timestamp: float
    hand: Observed
    tile_boxes: list[tuple[int, int, int, int]]
    stable: bool
    issues: list[str]
    schema_version: int = 1
    coordinate_space: str = 'normalized_game_1920x1080'
    mode: Observed = field(default_factory=Observed)
    self_seat: Observed = field(default_factory=Observed)
    seat_mapping: Observed = field(default_factory=Observed)
    round_info: Observed = field(default_factory=Observed)
    dora_indicators: Observed = field(default_factory=Observed)
    drawn_tile: Observed = field(default_factory=Observed)
    current_actor: Observed = field(default_factory=Observed)
    remaining_tiles: Observed = field(default_factory=Observed)
    legal_actions: Observed = field(default_factory=Observed)
    players: list[Player] = field(default_factory=lambda: [Player(x) for x in ('self', 'right', 'across', 'left')])
    history_complete: bool = False
    model_ready: bool = False

    def to_dict(self):
        return asdict(self)

class Classifier(Protocol):
    def __call__(self, bgr: np.ndarray) -> str: ...

class UnknownClassifier:
    def __call__(self, bgr):
        return '?'

class WrapperClassifier:
    """Load ONLY action/classifier.py; never import action/action.py or package init.

    Upstream returns argmax only. Repeated labels are not calibrated confidence.
    Use only a trusted local wrapper checkout (loading Python executes its code).
    """
    def __init__(self, checkout):
        path = Path(checkout).resolve() / 'action' / 'classifier.py'
        spec = importlib.util.spec_from_file_location('_visual_tile_classifier', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.classifier = module.Classify()

    def __call__(self, bgr):
        return self.classifier(bgr)

def normalize(frame, rect):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image')
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('Game rectangle outside screenshot')
    if abs(w/h - 16/9) > .03:
        raise ValueError('Game rectangle must match 16:9 content, excluding browser chrome')
    return cv2.resize(frame[y:y+h, x:x+w], (1920, 1080))

def hand_boxes(bgr):
    """Adapted from majsoul_wrapper GUIInterface._getHandTiles (MIT, Yue Qin).

    Same scan start, white threshold, fixed-range floodfill and size threshold.
    Bound scan and work ROI to prevent crashes/huge background regions.
    No mouse calls, screenshot globals, sleep, homography or SDK imports.
    """
    image = bgr.copy()
    # Ignore everything outside the hand band. Rectangles remain normalized.
    image[:830] = 0
    result = []
    x, y, fail = 235, 1002, 0
    while x < 1920 and fail < 100:
        if np.all(image[y, x] > 110):
            image[y, x] = 110
            _, _, _, (bx, by, w, h) = cv2.floodFill(image, None, (x, y), (0, 0, 0),
                (0, 0, 0), (145, 145, 145), flags=cv2.FLOODFILL_FIXED_RANGE)
            if 66 < w < 150 and 106 < h < 220:
                result.append((bx, by, w, h))
                fail = 0
                x = max(x, bx+w-1)
            else:
                fail += 1
        else:
            fail += 1
        x += 1
    return result

class Collector:
    def __init__(self, classifier: Classifier, stable_frames=3, max_gap=.8):
        if stable_frames < 2 or max_gap <= 0:
            raise ValueError('Need >=2 stable frames and positive max_gap')
        self.classifier, self.required, self.max_gap = classifier, stable_frames, max_gap
        self.last_time = None
        self.candidate = None
        self.count = 0
        self.frame_id = 0

    def observe(self, frame, rect, timestamp):
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('Timestamps must be finite and strictly increasing')
        gap = self.last_time is not None and timestamp-self.last_time > self.max_gap
        self.last_time = timestamp
        self.frame_id += 1
        bgr = normalize(frame, rect)
        boxes = hand_boxes(bgr)
        tiles = [self.classifier(bgr[y:y+h, x:x+w]) for x, y, w, h in boxes]
        issues = []
        if not 1 <= len(tiles) <= 14:
            issues.append('hand_not_detected_or_invalid_count')
        if any(t not in VALID for t in tiles):
            issues.append('unrecognized_tile')
        counts = Counter('5'+t[1] if t.startswith('0') else t for t in tiles)
        if any(n > 4 for n in counts.values()):
            issues.append('impossible_tile_multiplicity')
        if gap:
            issues.append('capture_gap_history_unknown')
        # Layout change is a change too (sorted hand alone cannot identify drawn tile).
        key = (tuple(tiles), tuple(boxes))
        self.count = self.count+1 if key == self.candidate and not gap and not issues else (1 if not issues else 0)
        self.candidate = key
        stable = self.count >= self.required
        return State(self.frame_id, timestamp,
                     Observed(tiles, 'screen_classifier', self.frame_id), boxes, stable, issues)

class MjaiBridge:
    """Extension seam. Hand-only observations cannot establish an event history."""
    def events(self, state: State) -> list[dict]:
        raise RuntimeError('MJAI reconstruction not implemented: need round start, seat mapping and verified ordered public events')

class MortalPort(Protocol):
    # Existing BotMortalLocal: init_bot(seat, GameMode.MJ4P), then react_batch(events).
    def react_batch(self, events: list[dict]) -> dict | None: ...

class HudPort(Protocol):
    # Existing GameBrowser method; do not construct BotManager to access it.
    def overlay_update_guidance(self, guide: str, subtitle: str, options: list): ...
    def overlay_clear_guidance(self): ...

def publish_observation(hud: HudPort, state: State):
    hud.overlay_clear_guidance()  # Invalidate old model advice on every new frame.
    text = ' '.join(state.hand.value) if state.hand.value else '未识别'
    hud.overlay_update_guidance(f'手牌: {text}', '视觉采集 / 无模型建议' + (' / 稳定' if state.stable else ' / 待确认'), [])
