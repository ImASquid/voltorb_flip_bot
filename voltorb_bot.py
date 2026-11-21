#!/usr/bin/env python3
"""
Voltorb Flip Bot
Modes:
  python voltorb_bot.py --mode sim --games 1000
  python voltorb_bot.py --mode manual
  python voltorb_bot.py --mode screen --calibrate
  python voltorb_bot.py --mode screen --read-only
  python voltorb_bot.py --mode screen --levels 8

Screen mode uses PyAutoGUI screenshots and mouse clicks. Move the mouse to a
screen corner to abort (PyAutoGUI fail-safe).
"""
from __future__ import annotations

import argparse
import itertools
import json
import operator
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

N = 5
CFG_PATH = Path("voltorb_config.json")
TPL_PATH = Path("voltorb_templates.json")
DEBUG_PATH = Path("voltorb_debug_grid.png")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class Inconsistent(Exception):
    pass


@lru_cache(maxsize=None)
def _row_options(points: int, volts: int, known: tuple):
    choices = [[v] if v is not None else [0, 1, 2, 3] for v in known]
    result = []
    for combo in itertools.product(*choices):
        if sum(combo) == points and combo.count(0) == volts:
            delta = tuple(combo) + tuple(int(v == 0) for v in combo)
            result.append((combo, delta))
    return result


def _valid(remaining, rows_left):
    for j in range(N):
        points, volts = remaining[j], remaining[N + j]
        if volts < 0 or volts > rows_left:
            return False
        non_volts = rows_left - volts
        if points < non_volts or points > 3 * non_volts:
            return False
    return True


def analyze(row_clues, col_clues, board):
    """Return exact per-cell probabilities [P(Voltorb), P(1), P(2), P(3)]."""
    options = []
    for i in range(N):
        row_options = _row_options(
            int(row_clues[i][0]), int(row_clues[i][1]), tuple(board[i])
        )
        if not row_options:
            raise Inconsistent(f"row {i + 1} has no valid arrangement")
        options.append(row_options)

    start = tuple(int(c[0]) for c in col_clues) + tuple(int(c[1]) for c in col_clues)
    zero = (0,) * (2 * N)
    if not _valid(start, N):
        raise Inconsistent("column clues are impossible")

    forward = [{start: 1}]
    for i in range(N):
        rows_left = N - 1 - i
        next_layer = {}
        for state, count in forward[i].items():
            for _, delta in options[i]:
                nxt = tuple(map(operator.sub, state, delta))
                if _valid(nxt, rows_left):
                    next_layer[nxt] = next_layer.get(nxt, 0) + count
        forward.append(next_layer)

    total = forward[N].get(zero, 0)
    if total == 0:
        raise Inconsistent("no board satisfies all clues")

    backward = [None] * (N + 1)
    backward[N] = {zero: 1}
    for i in range(N - 1, -1, -1):
        layer = {}
        for state in forward[i]:
            ways = 0
            for _, delta in options[i]:
                ways += backward[i + 1].get(
                    tuple(map(operator.sub, state, delta)), 0
                )
            if ways:
                layer[state] = ways
        backward[i] = layer

    counts = np.zeros((N, N, 4), dtype=np.float64)
    for i in range(N):
        for state, prefix_count in forward[i].items():
            if state not in backward[i]:
                continue
            for combo, delta in options[i]:
                suffix_count = backward[i + 1].get(
                    tuple(map(operator.sub, state, delta)), 0
                )
                if suffix_count:
                    weight = prefix_count * suffix_count
                    for j, value in enumerate(combo):
                        counts[i, j, value] += weight

    return counts / total


def plan(probabilities, board):
    """Choose guaranteed-safe multipliers first; otherwise choose lowest risk."""
    eps = 1e-12
    safe, candidates = [], []
    for i in range(N):
        for j in range(N):
            if board[i][j] is not None:
                continue
            p_volt = probabilities[i, j, 0]
            p_multiplier = probabilities[i, j, 2] + probabilities[i, j, 3]
            if p_multiplier > eps:
                candidates.append((p_volt, -p_multiplier, i, j))
                if p_volt < eps:
                    safe.append((-p_multiplier, i, j))
    if not candidates:
        return "done", []
    if safe:
        return "safe", [(i, j) for _, i, j in sorted(safe)]
    candidates.sort()
    return "guess", [(candidates[0][2], candidates[0][3])]


# ---------------------------------------------------------------------------
# Simulator and manual mode
# ---------------------------------------------------------------------------
def clues_from_board(board):
    rows = [(sum(row), row.count(0)) for row in board]
    cols = [
        (sum(board[i][j] for i in range(N)),
         sum(board[i][j] == 0 for i in range(N)))
        for j in range(N)
    ]
    return rows, cols


def random_board(rng):
    volts = rng.randint(6, 10)
    multipliers = rng.randint(4, 9)
    cells = [0] * volts
    cells += [rng.choice([2, 3]) for _ in range(multipliers)]
    cells += [1] * (25 - len(cells))
    rng.shuffle(cells)
    return [cells[i * N:(i + 1) * N] for i in range(N)]


class SimGame:
    def __init__(self, rng):
        self.board = random_board(rng)

    def read_clues(self):
        return clues_from_board(self.board)

    def flip(self, row, col):
        return self.board[row][col]


def parse_clues(line):
    values = line.replace(",", " ").split()
    if len(values) != 10:
        raise ValueError("enter exactly 10 numbers: points/volts for 5 lines")
    nums = list(map(int, values))
    pairs = [(nums[i], nums[i + 1]) for i in range(0, 10, 2)]
    for points, volts in pairs:
        if not (0 <= points <= 15 and 0 <= volts <= 5):
            raise ValueError("clue values are outside the valid range")
    return pairs


def manual_clues():
    print("Enter five points/Voltorb pairs, e.g. 5 1  4 2  7 0  3 3  6 1")
    while True:
        try:
            rows = parse_clues(input("ROW clues (top to bottom): "))
            cols = parse_clues(input("COLUMN clues (left to right): "))
            if (sum(x[0] for x in rows) == sum(x[0] for x in cols)
                    and sum(x[1] for x in rows) == sum(x[1] for x in cols)):
                return rows, cols
            print("Row and column totals disagree. Please re-enter both lines.")
        except ValueError as exc:
            print("Invalid input:", exc)


class ManualGame:
    def read_clues(self):
        return manual_clues()

    def flip(self, row, col):
        while True:
            value = input(
                f"Flip row {row + 1}, column {col + 1}. "
                "Value (0=Voltorb, 1, 2, 3): "
            ).strip()
            if value in {"0", "1", "2", "3"}:
                return int(value)


def play_level(game, verbose=True):
    rows, cols = game.read_clues()
    board = [[None] * N for _ in range(N)]
    guesses = 0
    while True:
        probabilities = analyze(rows, cols, board)
        kind, cells = plan(probabilities, board)
        if kind == "done":
            return True, guesses
        row, col = cells[0]
        if kind == "guess":
            guesses += 1
            if verbose:
                print(
                    f"GUESS row {row + 1}, col {col + 1}: "
                    f"P(Voltorb)={probabilities[row, col, 0]:.1%}, "
                    f"P(2 or 3)={probabilities[row, col, 2] + probabilities[row, col, 3]:.1%}"
                )
        elif verbose:
            print(f"SAFE row {row + 1}, col {col + 1}")
        value = game.flip(row, col)
        if value is None:
            return True, guesses
        board[row][col] = value
        if value == 0:
            return False, guesses


def run_sim(games, seed):
    rng = random.Random(seed)
    wins = guesses = 0
    start = time.time()
    for _ in range(games):
        won, count = play_level(SimGame(rng), verbose=False)
        wins += int(won)
        guesses += count
    elapsed = time.time() - start
    print(
        f"{games} simulated boards: {wins} wins ({wins / games:.1%}); "
        f"{guesses / games:.2f} forced guesses/board; "
        f"{elapsed / games * 1000:.1f} ms/board"
    )


# ---------------------------------------------------------------------------
# Screen recognition
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "bbox": None,
    "layout": {
        "pts_tens": [0.39, 0.12, 0.64, 0.40],
        "pts_ones": [0.64, 0.12, 0.90, 0.40],
        "volts": [0.64, 0.50, 0.92, 0.84],
        "card_margin": 0.18
    },
    "advance_keys": ["x"],
    "advance_delay": 0.8,
    "advance_max": 40,
    "diagnostics": True
}
GLYPH_SIZE = (10, 14)
CARD_SIZE = (12, 12)
DIGIT_THRESH = 0.035
CARD_THRESH = 0.07
DARK = 110
HIDDEN_COLORS = [(24, 132, 99), (41, 165, 107)]   # face-down card checkerboard
VOLTORB_RED = (231, 115, 82)                       # red cap of the Voltorb sprite
CARD_GLYPH_THRESH = 0.10


def load_json(path, default):
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(default, dict):
            merged = json.loads(json.dumps(default))
            for key, value in loaded.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            return merged
        return loaded
    return json.loads(json.dumps(default))


class Templates:
    def __init__(self, path=TPL_PATH):
        self.path = path
        self.data = load_json(path, {"digits": {}, "cards": {}, "cardglyphs": {}})

    def nearest(self, kind, vector):
        best_distance, best_label = 9.0, None
        for label, vectors in self.data[kind].items():
            for stored in vectors:
                stored = np.asarray(stored, dtype=np.float32)
                if kind in ("digits", "cardglyphs"):
                    distance = float(
                        np.abs(stored - vector).sum()
                        / (stored.sum() + vector.sum() + 1e-6)
                    )
                else:
                    distance = float(np.abs(stored - vector).mean())
                if distance < best_distance:
                    best_distance, best_label = distance, label
        return best_label, best_distance

    def add(self, kind, label, vector):
        self.data[kind].setdefault(str(label), []).append(
            np.round(vector, 3).tolist()
        )
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def has(self, kind, label):
        return bool(self.data[kind].get(str(label)))


def glyph_vec(crop):
    from PIL import Image
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    mask = arr.max(axis=2) < DARK
    ys, xs = np.where(mask)
    if len(ys) < 15:
        return np.zeros(GLYPH_SIZE[::-1], dtype=np.float32)
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    resized = Image.fromarray(box).resize(GLYPH_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def card_vec(tile, margin):
    from PIL import Image
    width, height = tile.size
    crop = tile.crop((
        int(width * margin), int(height * margin),
        int(width * (1 - margin)), int(height * (1 - margin))
    ))
    resized = crop.convert("RGB").resize(CARD_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).ravel() / 255.0


def ascii_glyph(vector):
    return "\n".join(
        "".join("#" if pixel > 0.5 else "." for pixel in row)
        for row in vector
    )


def ascii_card(tile):
    from PIL import Image
    gray = np.asarray(tile.convert("L").resize((28, 12)), dtype=np.float32) / 255.0
    ramp = " .:-=+*#%@"
    return "\n".join(
        "".join(ramp[min(9, int(pixel * 10))] for pixel in row)
        for row in gray
    )


class ScreenGame:
    def __init__(self, cfg, templates, flip_delay=0.6):
        import pyautogui
        self.pyautogui = pyautogui
        self.cfg = cfg
        self.templates = templates
        self.flip_delay = flip_delay
        self.scale = 1.0
        if not cfg.get("bbox"):
            raise RuntimeError("No calibration found. Run --mode screen --calibrate first.")

    def grab(self):
        image = self.pyautogui.screenshot().convert("RGB")
        self.scale = image.width / self.pyautogui.size().width
        return image

    def tile(self, image, row, col):
        x0, y0, x1, y1 = self.cfg["bbox"]
        tile_w, tile_h = (x1 - x0) / 6, (y1 - y0) / 6
        return image.crop((
            int(x0 + col * tile_w), int(y0 + row * tile_h),
            int(x0 + (col + 1) * tile_w), int(y0 + (row + 1) * tile_h)
        ))

    def center(self, row, col):
        x0, y0, x1, y1 = self.cfg["bbox"]
        return (
            x0 + (col + 0.5) * (x1 - x0) / 6,
            y0 + (row + 0.5) * (y1 - y0) / 6
        )

    @staticmethod
    def sub(tile, fractions):
        width, height = tile.size
        return tile.crop((
            int(fractions[0] * width), int(fractions[1] * height),
            int(fractions[2] * width), int(fractions[3] * height)
        ))

    def ask(self, kind, context, art, choices):
        print("\n" + art)
        while True:
            answer = input(
                f"Unrecognized {kind} [{context}]. "
                f"Enter {'/'.join(choices)}: "
            ).strip().lower()
            if answer in choices:
                return answer

    def read_digit(self, tile, slot, context):
        vector = glyph_vec(self.sub(tile, self.cfg["layout"][slot]))
        if vector.sum() == 0:
            if slot == "pts_tens":
                return 0
            raise Inconsistent(
                f"{context} {slot}: crop is blank, so the grid calibration is "
                "probably off. Re-run --calibrate and check the debug grid."
            )

        label, distance = self.templates.nearest("digits", vector)
        if self.cfg.get("diagnostics", True):
            print(
                f"  OCR {context} {slot}: detected={label!r}, "
                f"distance={distance:.4f}, threshold={DIGIT_THRESH:.3f}"
            )

        if label is None or distance > DIGIT_THRESH:
            label = self.ask(
                "digit", f"{context} {slot}", ascii_glyph(vector),
                [str(number) for number in range(10)]
            )
            self.templates.add("digits", label, vector)
        return int(label)

    def read_clue(self, image, row, col, context):
        tile = self.tile(image, row, col)
        points = (
            10 * self.read_digit(tile, "pts_tens", context)
            + self.read_digit(tile, "pts_ones", context)
        )
        volts = self.read_digit(tile, "volts", context)
        if points > 15 or volts > 5:
            raise Inconsistent(
                f"{context} OCR produced impossible clue ({points}, {volts}); "
                "check the debug grid and clue crop settings"
            )
        return points, volts

    def read_clues(self):
        image = self.grab()
        rows = [
            self.read_clue(image, row, 5, f"row {row + 1}")
            for row in range(N)
        ]
        cols = [
            self.read_clue(image, 5, col, f"column {col + 1}")
            for col in range(N)
        ]
        print("\nRecognized clues:")
        print("  rows:", rows)
        print("  cols:", cols)

        row_points = sum(x[0] for x in rows)
        col_points = sum(x[0] for x in cols)
        row_volts = sum(x[1] for x in rows)
        col_volts = sum(x[1] for x in cols)
        if row_points != col_points or row_volts != col_volts:
            print(
                "\nCLUE CHECK FAILED:"
                f"\n  row point total={row_points}; column point total={col_points}"
                f"\n  row Voltorb total={row_volts}; column Voltorb total={col_volts}"
                "\nDo not automate this board. Correct the OCR templates/crops or use manual mode."
            )
            raise Inconsistent("row and column clue totals disagree")
        return rows, cols

    def click(self, row, col):
        """Move, pause, press, hold, release.

        The emulator reads touch input once per frame, so an instant
        press/release (PyAutoGUI's default click) is often missed.
        """
        x, y = self.center(row, col)
        x, y = x / self.scale, y / self.scale
        self.pyautogui.moveTo(x, y, duration=0.08)
        self.pyautogui.doubleClick()
        time.sleep(0.16)
        print(f"Clicking card ({row + 1}, {col + 1})")
        self.pyautogui.doubleClick()
        print(f"Click Attempt on ({row + 1}, {col + 1}) completed")

    def classify_card(self, tile):
        """Return (label, glyph_vector); label is 'h', '0'-'3', or None if unknown."""
        arr = np.asarray(tile.convert("RGB"), dtype=np.int16)
        height, width = arr.shape[:2]
        core = arr[int(height * 0.25):int(height * 0.75),
                   int(width * 0.25):int(width * 0.75)]

        def share(color, tol=14):
            return float((np.abs(core - np.array(color)).max(axis=2) <= tol).mean())

        if sum(share(color) for color in HIDDEN_COLORS) > 0.5:
            return "h", None
        if share(VOLTORB_RED) > 0.06:
            return "0", None
        vector = glyph_vec(self.sub(tile, [0.25, 0.25, 0.75, 0.75]))
        if vector.sum() == 0:
            return None, None
        label, distance = self.templates.nearest("cardglyphs", vector)
        if label is not None and distance <= CARD_GLYPH_THRESH:
            return label, vector
        return None, vector

    def flip(self, row, col):
        context = f"card ({row + 1},{col + 1})"
        attempts = 4
        for attempt in range(attempts):
            self.click(row, col)
            time.sleep(self.flip_delay)
            start = time.time()
            deadline = start + 7
            last, agree = None, 0
            prev_vec, still = None, 0
            left_hidden = False
            while time.time() < deadline:
                tile = self.tile(self.grab(), row, col)
                label, vector = self.classify_card(tile)
                if label != "h":
                    left_hidden = True
                elif not left_hidden and time.time() - start > 2.5:
                    break      # nothing happened: the click was missed, retry now
                if label in {"0", "1", "2", "3"}:
                    agree = agree + 1 if label == last else 1
                    last = label
                    if agree >= 2:
                        return int(label)
                else:
                    last, agree = None, 0
                # A revealed card the recognizer doesn't know: once it stops
                # changing, stop waiting and ask.
                if label is None and vector is not None:
                    unchanged = (
                        prev_vec is not None
                        and float(np.abs(vector - prev_vec).mean()) < 0.02
                    )
                    still = still + 1 if unchanged else 0
                    prev_vec = vector
                    if still >= 3:
                        break
                else:
                    prev_vec, still = None, 0
                time.sleep(0.15)

            tile = self.tile(self.grab(), row, col)
            label, vector = self.classify_card(tile)
            if label in {"0", "1", "2", "3"}:
                return int(label)
            if label is None and vector is not None:
                answer = self.ask(
                    "card", context, ascii_card(tile), ["0", "1", "2", "3"]
                )
                if answer != "0":
                    self.templates.add("cardglyphs", answer, vector)
                return int(answer)
            tile.save("voltorb_stuck_card.png")
            print(
                f"{context}: no flip detected (classifier saw {label!r}); "
                f"retrying click ({attempt + 1}/{attempts}). "
                "Saved voltorb_stuck_card.png"
            )
        raise RuntimeError(
            f"Card ({row + 1}, {col + 1}) did not flip. Check calibration/focus; "
            "see voltorb_stuck_card.png."
        )


def draw_debug(cfg):
    import pyautogui
    from PIL import ImageDraw
    image = pyautogui.screenshot().convert("RGB")
    if not cfg.get("bbox"):
        raise RuntimeError("No saved calibration. Run --calibrate first.")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = cfg["bbox"]
    tw, th = (x1 - x0) / 6, (y1 - y0) / 6
    for row in range(6):
        for col in range(6):
            tx, ty = x0 + col * tw, y0 + row * th
            draw.rectangle([tx, ty, tx + tw, ty + th], outline=(255, 0, 0), width=2)
            if row == 5 or col == 5:
                for slot, color in (
                    ("pts_tens", (0, 255, 255)),
                    ("pts_ones", (255, 255, 0)),
                    ("volts", (255, 0, 255))
                ):
                    f = cfg["layout"][slot]
                    draw.rectangle([
                        tx + f[0] * tw, ty + f[1] * th,
                        tx + f[2] * tw, ty + f[3] * th
                    ], outline=color, width=2)
    image.save(DEBUG_PATH)
    print(f"Saved {DEBUG_PATH.resolve()}")


BAND_COLORS = [
    (231, 115, 82), (66, 173, 66), (239, 165, 57), (49, 148, 255), (198, 99, 231)
]


def _run(profile, low=0.05, major=0.25, bridge=0.3):
    """Extent (start, end) of the heaviest group of nearby runs in a profile.

    Digits and icons punch small gaps into the clue bands, so runs of
    significant mass separated by gaps of up to `bridge` x their width are
    merged. Small isolated features (connectors) are ignored.
    """
    if profile.max() <= 0:
        return None
    on = profile >= low * profile.max()
    runs, start = [], None
    for i, flag in enumerate(list(on) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i, float(profile[start:i].sum())))
            start = None
    if not runs:
        return None
    top = max(r[2] for r in runs)
    runs = [r for r in runs if r[2] >= major * top]
    groups = [[runs[0]]]
    for run in runs[1:]:
        prev = groups[-1][-1]
        gap = run[0] - prev[1]
        if gap <= bridge * max(run[1] - run[0], prev[1] - prev[0]):
            groups[-1].append(run)
        else:
            groups.append([run])
    best = max(groups, key=lambda g: sum(r[2] for r in g))
    return best[0][0], best[-1][1]


def auto_bbox(image):
    """Locate the 6x6 tile grid from the five colored clue bands.

    The band colors are fixed by the game and every band is centered on its
    tile, so their positions give the grid origin and pitch without any mouse
    hovering. Returns [x0, y0, x1, y1] or raises ValueError.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    masks = [np.abs(arr - np.array(c)).max(axis=2) <= 6 for c in BAND_COLORS]
    everything = np.any(masks, axis=0)
    if everything.sum() < 500:
        raise ValueError("clue-band colors not found on screen")

    col_run = _run(everything.sum(axis=0))   # x-range of the row-clue column
    row_run = _run(everything.sum(axis=1))   # y-range of the column-clue row
    if col_run is None or row_run is None:
        raise ValueError("could not find the clue column/row")

    ys, xs = [], []
    for k, mask in enumerate(masks):
        span_y = _run(mask[:, col_run[0]:col_run[1]].sum(axis=1))
        span_x = _run(mask[row_run[0]:row_run[1], :].sum(axis=0))
        if span_y is None or span_x is None:
            raise ValueError(f"clue band {k + 1} not found")
        ys.append((span_y[0] + span_y[1]) / 2)
        xs.append((span_x[0] + span_x[1]) / 2)
    xs.append((col_run[0] + col_run[1]) / 2)

    def fit(values):
        """Least-squares line through tile centers, dropping one outlier."""
        idx = list(range(len(values)))
        for _ in range(2):
            slope, icpt = np.polyfit(idx, [values[i] for i in idx], 1)
            resid = [abs(icpt + slope * i - values[i]) for i in idx]
            if max(resid) <= 0.06 * slope:
                return slope, icpt
            idx.pop(int(np.argmax(resid)))
        raise ValueError("clue bands are not on a regular grid")

    px, ax = fit(xs)
    py, ay = fit(ys)
    if px < 20 or abs(px - py) > 0.04 * px:
        raise ValueError(f"inconsistent tile pitch ({px:.1f} x {py:.1f})")
    return [ax - px / 2, ay - py / 2, ax + 5.5 * px, ay + 5.5 * py]


def calibrate():
    import pyautogui
    cfg = load_json(CFG_PATH, DEFAULT_CFG)
    print("Make sure a fresh Voltorb Flip board is fully visible.")
    input("Press Enter to auto-detect the grid...")
    image = pyautogui.screenshot().convert("RGB")
    try:
        cfg["bbox"] = [round(v, 2) for v in auto_bbox(image)]
        print("Auto-detected grid:", cfg["bbox"])
    except ValueError as exc:
        print(f"Auto-detect failed ({exc}). Falling back to manual calibration.")
        print(
            "  [1][ ][ ][ ][ ] | clue\n"
            "  [ ][ ][ ][ ][ ] | clue\n"
            "  [ ][ ][ ][ ][ ] | clue\n"
            "  [ ][ ][ ][ ][ ] | clue\n"
            "  [ ][ ][ ][ ][2] | clue      <- 1 and 2 are the two CARDS to hover\n"
            "  clue clue clue clue clue    (NOT the empty bottom-right corner)"
        )
        input("Hover the CENTER of card 1 (top-left card), press Enter...")
        first = pyautogui.position()
        input("Hover the CENTER of card 2 (the 5th card in the 5th row), press Enter...")
        last = pyautogui.position()
        scale = image.width / pyautogui.size().width
        px = (last.x - first.x) * scale / 4
        py = (last.y - first.y) * scale / 4
        x0, y0 = first.x * scale - px / 2, first.y * scale - py / 2
        cfg["bbox"] = [x0, y0, x0 + 6 * px, y0 + 6 * py]
    CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    draw_debug(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sim", "manual", "screen"], default="manual")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--flip-delay", type=float, default=0.6)
    parser.add_argument("--reset-templates", action="store_true")
    parser.add_argument("--debug-grid", action="store_true")
    args = parser.parse_args()

    if args.mode == "sim":
        if args.games < 1:
            parser.error("--games must be at least 1")
        run_sim(args.games, args.seed)
        return

    if args.reset_templates and TPL_PATH.exists():
        TPL_PATH.unlink()
        print("Deleted saved templates.")

    if args.mode == "manual":
        game = ManualGame()
    else:
        if args.calibrate:
            calibrate()
            return
        cfg = load_json(CFG_PATH, DEFAULT_CFG)
        if args.debug_grid:
            draw_debug(cfg)
            return
        game = ScreenGame(cfg, Templates(), args.flip_delay)
        if args.read_only:
            game.read_clues()
            print("\nRead-only test complete. No cards were clicked.")
            return

    level = wins = 0
    while args.levels == 0 or level < args.levels:
        level += 1
        print(f"\n=== Level {level} ===")
        try:
            won, guesses = play_level(game)
        except (Inconsistent, RuntimeError) as exc:
            sys.exit(f"\nStopped safely: {exc}")
        wins += int(won)
        print(
            f"Result: {'WON' if won else 'Voltorb hit'}; "
            f"{guesses} guesses; record {wins}/{level}"
        )
        if args.mode == "screen" and (args.levels == 0 or level < args.levels):
            print("Waiting for the next board transition...")
            time.sleep(2.0)

            print("Advancing to next board...")
            for i in range(12):
                game.pyautogui.doubleClick(1964, 767)
                time.sleep(0.8)

            print("Waiting for the game to finish loading...")
            time.sleep(2.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped by user.")