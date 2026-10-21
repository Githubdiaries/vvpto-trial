#!/usr/bin/env python3
"""
bbb_display.py - runs on the BEAGLEBONE. Receives vein masks from the Pi and
draws them on the DLP projector: black background, veins in a solid colour.
Black = no light, so only the veins are lit on the skin.

Usage
    python3 bbb_display.py                       # fullscreen OpenCV window
    python3 bbb_display.py --fb /dev/fb0         # draw straight to a framebuffer
    python3 bbb_display.py --test-pattern        # alignment grid, no Pi needed

Alignment keys (window mode only):
    j / l : shift left / right     i / k : shift up / down
    + / - : scale up / down        h / v : mirror horizontally / vertically
    p     : print the current values (copy them into config.py)   q : quit
"""
import argparse
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import NET, PROJECTOR, VISION
from transport import FrameReceiver


# ==========================================================================
# Geometry: camera mask -> projector pixels
# ==========================================================================
def build_transform(src: Tuple[int, int], dst: Tuple[int, int], flip_h: bool,
                    flip_v: bool, scale: float, off_x: int, off_y: int) -> np.ndarray:
    """
    2x3 affine matrix mapping a src-sized mask onto a dst-sized projector frame.
    Step 1 stretches the mask to fill the projector; step 2 applies the user's
    flip / scale (about the projector centre) and shift. This is what lines the
    projected veins up with the real veins on the skin.
    """
    sw, sh = src
    dw, dh = dst
    base = np.array([[dw / sw, 0, 0], [0, dh / sh, 0], [0, 0, 1]], np.float64)

    cx, cy = dw / 2.0, dh / 2.0
    fx = -scale if flip_h else scale
    fy = -scale if flip_v else scale
    adjust = np.array([[fx, 0, cx - fx * cx + off_x],
                       [0, fy, cy - fy * cy + off_y],
                       [0, 0, 1]], np.float64)
    return (adjust @ base)[:2]


def render(mask: np.ndarray, matrix: np.ndarray, size: Tuple[int, int],
           color_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Warp the mask to projector size and colour it. Returns BGR uint8."""
    warped = cv2.warpAffine(mask, matrix, size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    planes = [(warped * (c / 255.0)).astype(np.uint8) for c in color_bgr]
    return cv2.merge(planes)


def test_pattern(size: Tuple[int, int]) -> np.ndarray:
    """
    Alignment grid drawn in CAMERA coordinates: its border is the edge of the
    camera's field of view and the cross marks its centre. When the projected
    border/cross land where the camera really sees its edge/centre, the
    projector and camera are aligned.
    """
    w, h = size
    pat = np.zeros((h, w), np.uint8)
    cv2.rectangle(pat, (1, 1), (w - 2, h - 2), 255, 2)
    for i in (1, 2):                                   # rule-of-thirds grid
        cv2.line(pat, (w * i // 3, 0), (w * i // 3, h), 120, 1)
        cv2.line(pat, (0, h * i // 3), (w, h * i // 3), 120, 1)
    cv2.line(pat, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), 255, 2)
    cv2.line(pat, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), 255, 2)
    return pat


# ==========================================================================
# Output back-ends
# ==========================================================================
class WindowOutput:
    """Fullscreen OpenCV window (the projector is the second/only display)."""

    name = "veinscope"

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, bgr: np.ndarray) -> int:
        cv2.imshow(self.name, bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()


class FramebufferOutput:
    """Writes pixels straight into /dev/fbN. No desktop/X server needed."""

    def __init__(self, device: str) -> None:
        sysfs = f"/sys/class/graphics/{os.path.basename(device)}"
        w, h = (int(v) for v in open(f"{sysfs}/virtual_size").read().split(","))
        self.size = (w, h)
        self._bpp = int(open(f"{sysfs}/bits_per_pixel").read())
        self._stride = int(open(f"{sysfs}/stride").read())     # bytes per row
        if self._bpp not in (16, 32):
            raise SystemExit(f"Unsupported framebuffer depth: {self._bpp} bpp")
        self._fb = open(device, "r+b", buffering=0)

    def show(self, bgr: np.ndarray) -> int:
        if self._bpp == 32:
            px = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        else:
            px = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGR565)
        h, w = px.shape[:2]
        row_bytes = w * self._bpp // 8
        rows = px.reshape(h, row_bytes)
        if self._stride != row_bytes:                          # pad each row
            padded = np.zeros((h, self._stride), np.uint8)
            padded[:, :row_bytes] = rows
            rows = padded
        self._fb.seek(0)
        self._fb.write(rows.tobytes())
        time.sleep(0.005)                                      # keep CPU polite
        return 255                                             # no keyboard here

    def close(self) -> None:
        self._fb.close()


# ==========================================================================
# Main
# ==========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fb", metavar="DEVICE", help="framebuffer device, e.g. /dev/fb0")
    p.add_argument("--test-pattern", action="store_true", help="show alignment grid")
    p.add_argument("--flip-h", action="store_true", default=PROJECTOR.flip_h)
    p.add_argument("--flip-v", action="store_true", default=PROJECTOR.flip_v)
    p.add_argument("--scale", type=float, default=PROJECTOR.scale)
    p.add_argument("--offset-x", type=int, default=PROJECTOR.offset_x)
    p.add_argument("--offset-y", type=int, default=PROJECTOR.offset_y)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = FramebufferOutput(args.fb) if args.fb else WindowOutput(PROJECTOR.size)
    receiver = None if args.test_pattern else FrameReceiver(NET.port)

    flip_h, flip_v = args.flip_h, args.flip_v
    scale, off_x, off_y = args.scale, args.offset_x, args.offset_y
    blank = np.zeros((out.size[1], out.size[0], 3), np.uint8)
    last_seq: Optional[int] = None
    frame = blank

    try:
        while True:
            matrix = build_transform(VISION.work_size, out.size,
                                     flip_h, flip_v, scale, off_x, off_y)
            if receiver is None:                                # test pattern
                frame = render(test_pattern(VISION.work_size), matrix,
                               out.size, PROJECTOR.color_bgr)
            else:
                item = receiver.latest()
                if item is None or time.time() - item.received_at > PROJECTOR.stale_seconds:
                    frame, last_seq = blank, None               # nothing fresh: stay dark
                elif item.seq != last_seq:                      # render new frames only
                    frame = render(item.mask, matrix, out.size, PROJECTOR.color_bgr)
                    last_seq = item.seq

            key = out.show(frame)

            # ---- live alignment (window mode) ----
            step = 2
            if key == ord("q"):
                break
            elif key == ord("j"): off_x -= step
            elif key == ord("l"): off_x += step
            elif key == ord("i"): off_y -= step
            elif key == ord("k"): off_y += step
            elif key in (ord("+"), ord("=")): scale *= 1.01
            elif key == ord("-"): scale /= 1.01
            elif key == ord("h"): flip_h = not flip_h
            elif key == ord("v"): flip_v = not flip_v
            elif key == ord("p"):
                print(f"flip_h={flip_h} flip_v={flip_v} scale={scale:.3f} "
                      f"offset_x={off_x} offset_y={off_y}")
            if key in (ord("+"), ord("="), ord("-"), ord("j"), ord("l"),
                       ord("i"), ord("k"), ord("h"), ord("v")):
                last_seq = None                                 # force re-render
    except KeyboardInterrupt:
        pass
    finally:
        if receiver is not None:
            receiver.close()
        out.close()


if __name__ == "__main__":
    main()
