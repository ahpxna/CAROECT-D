#!/usr/bin/env python3
"""
create_calib_pattern.py — Stage 1 (Eq. 30) two-level blinking calibration
pattern for a monitor, replacing the earlier checkerboard version
(create_gif.py).

WHY NOT THE OLD DESIGN
------------------------
1. Pure black/white (code values 0/255) as the two levels is wrong, but NOT
   for a "ratio" reason — the monitor's absolute brightness genuinely does
   NOT matter here, because ΔL_scene is measured empirically from the
   camera's own Linear-decoded TIFF capture of this pattern (paper
   Section III-D), not computed from the display's nominal code values or
   EOTF. The real problem is SNR: near-0 code values land in the camera's
   noise floor / shot-noise-dominated shadow region (see the Poisson/shot
   noise discussion already established for this project), so the
   measured ΔL at that level is noisy, not just "very negative". Near-255
   risks sensor clipping in the bright state. Two mid-gray levels keep
   both states comfortably inside the camera's well-exposed linear range.

2. A fine checkerboard mixes a SPATIAL contrast step (at every square
   boundary) into what should be a PURE TEMPORAL contrast step. Pixels
   near a boundary see both at once, contaminating the ΔL measurement
   Eq. 30 needs to isolate. A single large, spatially uniform block gives
   every interior pixel a clean, boundary-free temporal-only step.

3. 20000 loop iterations at 0.5 s/frame = ~5.6 hours and a huge file. Cut
   to a small, finite number of blink cycles, each held long enough to
   collect many camera frames per state at the project's 119.88 fps
   capture rate (see config.yaml fps_original) for burst averaging.

WHAT THIS SCRIPT DOES NOT DO
-------------------------------
It does NOT compute or assume ΔL_scene from these code values — that
measurement happens downstream, from the Nikon+DaVinci Linear TIFF capture
of this pattern (Eq. 30 in the paper), exactly as already established.
This script's only job is to present two stable, distinct, well-exposed
temporal states on screen for long enough to be captured cleanly by both
the RGB rig and the co-located Triton2 EVS.
"""

import cv2
import imageio
import numpy as np

# ── Pattern parameters ──────────────────────────────────────────────
width, height = 800, 800

# Two MID-GRAY levels, not black/white — see WHY NOT THE OLD DESIGN #1.
# Kept well clear of both 0 (shadow noise floor) and 255 (clipping).
LEVEL_DARK = 70
LEVEL_BRIGHT = 190

# Thin registration border in a THIRD gray value, distinct from both
# levels, so the test patch's boundary is identifiable in captured
# footage during analysis without touching the interior pixels used
# for the actual ΔL measurement.
BORDER_GRAY = 128
BORDER_PX = 20

# Camera captures at 119.88 fps (config.yaml fps_original) — hold each
# state long enough to collect many camera frames per state for burst
# averaging in Eq. 30 (N̄ over many crossings and pixels).
SECONDS_PER_STATE = 1.0
N_CYCLES = 20  # 20 dark->bright->dark cycles = 40 states total


def make_frame(level: int) -> np.ndarray:
    img = np.full((height, width, 3), level, dtype=np.uint8)
    img[:BORDER_PX, :] = BORDER_GRAY
    img[-BORDER_PX:, :] = BORDER_GRAY
    img[:, :BORDER_PX] = BORDER_GRAY
    img[:, -BORDER_PX:] = BORDER_GRAY
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


frame_dark = make_frame(LEVEL_DARK)
frame_bright = make_frame(LEVEL_BRIGHT)

frames = []
for _ in range(N_CYCLES):
    frames.append(frame_dark)
    frames.append(frame_bright)

# duration is per-frame display time in the saved GIF (seconds)
imageio.mimsave(
    "stage1_calib_pattern.gif",
    frames,
    duration=SECONDS_PER_STATE,
    loop=0,  # loop indefinitely when played on the monitor
)

total_s = len(frames) * SECONDS_PER_STATE
print(f"Saved stage1_calib_pattern.gif")
print(f"  {N_CYCLES} cycles, {len(frames)} states, "
      f"{SECONDS_PER_STATE}s/state, {total_s:.0f}s total")
print(f"  Levels: dark={LEVEL_DARK} bright={LEVEL_BRIGHT} "
      f"(mid-gray, avoids shadow noise floor and highlight clipping)")
print(f"\nNext: play this fullscreen on the calibration monitor, record")
print(f"with the RGB rig through the normal DaVinci Linear pipeline AND")
print(f"the co-located Triton2 simultaneously, then measure ΔL_scene from")
print(f"the resulting linear TIFFs (not from the code values above).")
