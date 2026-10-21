#!/usr/bin/env python3

import sys
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# Import the existing project's settings and vein detector.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import VISION
from vision import VeinPipeline

WIDTH, HEIGHT = 640, 480
FPS = 20

WORK_W, WORK_H = VISION.work_size
OUT_W, OUT_H = WORK_W * 4, WORK_H

RTSP_URL = "rtsp://127.0.0.1:8554/cam"


def add_label(image, label):
    cv2.putText(
        image, label, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 255, 255), 1, cv2.LINE_AA
    )
    return image


def main():
    camera = None
    camera_started = False
    encoder = None

    try:
        print("Opening Camera Module 3 NoIR...")

        camera = Picamera2()
        cam_config = camera.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "YUV420"},
            buffer_count=4,
            controls={
                "FrameDurationLimits": (
                    int(1_000_000 / FPS),
                    int(1_000_000 / FPS),
                ),
                "AfMode": controls.AfModeEnum.Continuous,
            },
        )
        camera.configure(cam_config)
        camera.start()
        camera_started = True
        time.sleep(1)

        # IR array is continuously powered: do not use LED lock-in.
        pipeline = VeinPipeline(VISION, pulse=False)

        print("Starting low-latency H.264 publisher...")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{OUT_W}x{OUT_H}",
            "-framerate", str(FPS),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-bf", "0",
            "-g", str(FPS),
            "-b:v", "1200k",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            RTSP_URL,
        ]

        encoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

        print("Processing frames. Open http://100.75.101.73:8889/cam")
        print("Press Ctrl+C to stop.")

        frames = 0
        started_at = time.monotonic()

        while True:
            yuv = camera.capture_array("main")
            luma = yuv[:HEIGHT, :WIDTH]

            # Existing vein pipeline: contrast, ridge detection and mask.
            mask = pipeline.push(luma)

            small = cv2.resize(
                luma, VISION.work_size,
                interpolation=cv2.INTER_AREA,
            )

            enhanced = pipeline.debug.get("enhanced")
            if enhanced is None or enhanced.shape != small.shape:
                enhanced = cv2.equalizeHist(small)

            if mask is None:
                mask = np.zeros((WORK_H, WORK_W), dtype=np.uint8)

            raw_panel = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            enhanced_panel = cv2.cvtColor(
                enhanced, cv2.COLOR_GRAY2BGR
            )
            mask_panel = cv2.cvtColor(
                mask, cv2.COLOR_GRAY2BGR
            )

            green = np.zeros_like(enhanced_panel)
            green[:, :, 1] = mask
            overlay = cv2.addWeighted(
                enhanced_panel, 0.85, green, 0.8, 0
            )

            panels = [
                add_label(raw_panel, "RAW"),
                add_label(enhanced_panel, "ENHANCED"),
                add_label(mask_panel, "VEIN MASK"),
                add_label(overlay, "VEIN OVERLAY"),
            ]

            output_frame = np.hstack(panels)

            if encoder.poll() is not None:
                raise RuntimeError(
                    f"FFmpeg exited with code {encoder.returncode}"
                )

            encoder.stdin.write(output_frame.tobytes())

            frames += 1
            now = time.monotonic()
            if now - started_at >= 2:
                print(
                    f"Processing rate: "
                    f"{frames / (now - started_at):.1f} FPS"
                )
                frames = 0
                started_at = now

    except KeyboardInterrupt:
        print("\nStopping processing stream...")
    finally:
        if encoder is not None:
            if encoder.stdin:
                try:
                    encoder.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            if encoder.poll() is None:
                encoder.terminate()

            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()

        if camera is not None:
            if camera_started:
                camera.stop()
            camera.close()

        print("Camera and processing stream stopped.")


if __name__ == "__main__":
    main()
