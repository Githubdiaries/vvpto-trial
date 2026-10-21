#!/usr/bin/env python3
"""
pi_main.py - runs on the RASPBERRY PI 4.

Loop:  grab frame -> flip IR LED -> vein vision -> send mask to the BeagleBone.

Usage
    python3 pi_main.py                # normal: stream masks to the BeagleBone
    python3 pi_main.py --preview      # also show debug windows (needs a display)
    python3 pi_main.py --no-net       # camera + vision only, no BeagleBone yet
Press Ctrl+C (or 'q' in the preview window) to stop.
"""
import argparse
import time

import numpy as np

from config import LED, NET, VISION
from hardware import IrLed, NoirCamera
from transport import FrameSender
from vision import VeinPipeline


def show_preview(pipeline: VeinPipeline) -> bool:
    """Display signal | enhanced | mask side by side. Returns False on 'q'."""
    import cv2

    if not pipeline.debug:
        return True
    panel = np.hstack([pipeline.debug["signal"],
                       pipeline.debug["enhanced"],
                       pipeline.debug["mask"]])
    cv2.imshow("veinscope: signal | enhanced | mask  (q = quit)", panel)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="show debug windows")
    parser.add_argument("--no-net", action="store_true", help="do not send to the BeagleBone")
    args = parser.parse_args()

    camera = NoirCamera()
    led = IrLed()
    pipeline = VeinPipeline(VISION, pulse=LED.pulse)
    sender = None if args.no_net else FrameSender(NET.bbb_ip, NET.port)

    frames, t0 = 0, time.time()
    try:
        while True:
            luma = camera.grab_luma()     # blocks until the next frame is ready

            # Exactly ONE LED flip per frame. The camera pipeline delays the
            # effect by a frame or two, but the on/off alternation is kept, and
            # vision.py finds out by itself which frame of a pair was "on".
            led.step()

            mask = pipeline.push(luma)
            if mask is None:              # still warming up
                continue
            if sender is not None:
                sender.submit(mask)
            if args.preview and not show_preview(pipeline):
                break

            # Status line every ~2 seconds.
            frames += 1
            if time.time() - t0 >= 2.0:
                link = "n/a" if sender is None else ("up" if sender.connected else "DOWN")
                led_state = "ok" if pipeline.led_ok else "NOT DETECTED"
                print(f"{frames / (time.time() - t0):5.1f} fps | "
                      f"led modulation: {led_state} | link: {link}")
                frames, t0 = 0, time.time()
    except KeyboardInterrupt:
        pass
    finally:                              # always leave hardware in a safe state
        if sender is not None:
            sender.close()
        led.close()
        camera.close()


if __name__ == "__main__":
    main()
