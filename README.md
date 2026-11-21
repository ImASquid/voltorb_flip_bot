# Voltorb Flip Bot

A Python bot for the Voltorb Flip minigame. It includes:
- Exact probability solver using dynamic programming.
- Random-board simulator.
- Manual clue-entry mode.
- Screen capture, clue OCR templates, and mouse clicking.
- Diagnostic output for every clue digit.
- A safety check that stops if row and column clue totals disagree.

## Requirements

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## 1. Test the solver first

```bash
python voltorb_bot.py --mode sim --games 1000
```

This does not interact with the game.

## 2. Screen setup

Open the game in your browser/emulator. Keep the game visible and unobstructed.
The screen mode assumes a 5x5 card grid with one clue column on the right and
one clue row along the bottom, making a 6x6 tile region.

Calibrate the outer corners:

```bash
python voltorb_bot.py --mode screen --calibrate
```

This creates `voltorb_config.json` and `voltorb_debug_grid.png`.
Open the debug image and confirm the red grid lines follow the tile boundaries.
The colored clue crop boxes should cover the relevant digits.

## 3. Train templates and read clues

Start on a fresh board with all 25 cards face-down:

```bash
python voltorb_bot.py --mode screen --read-only
```

The first run asks to learn the card back. Unknown digits are shown as ASCII
masks and labelled by you. Templates are saved to `voltorb_templates.json`.

The program prints each OCR result and its distance from the closest template.
If row and column clue totals do not agree, it stops rather than playing from
bad readings.

Important: a blank digit crop is interpreted as zero. If the diagnostics show
a clue digit as blank when it should not be, adjust the `pts_tens`, `pts_ones`,
or `volts` crop fractions in `voltorb_config.json`.

## 4. Full play

Only after read-only mode consistently reads the clues correctly:

```bash
python voltorb_bot.py --mode screen --levels 1
```

Start with one level. Increase `--levels` only after verifying clicking works.
Move the mouse to a screen corner to trigger PyAutoGUI's fail-safe.

## Manual mode

```bash
python voltorb_bot.py --mode manual
```

Enter five row clue pairs, then five column clue pairs. Each pair is:
`points voltorbs`.

## Files

- `voltorb_bot.py`: main application, solver, recognition, and automation.
- `voltorb_config.json`: screen calibration and crop layout.
- `voltorb_templates.json`: learned digit and card image templates.
- `requirements.txt`: Python dependencies.
- `run_sim.bat`: Windows shortcut for the simulation.
- `run_read_only.bat`: Windows shortcut for screen read-only mode.
- `run_screen.bat`: Windows shortcut for one-level automation.

The included config has no calibration coordinates. Run calibration on the
computer where the bot will be used.
