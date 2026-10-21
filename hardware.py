"""
hardware.py - everything that touches Raspberry Pi hardware.

    NoirCamera : Camera Module 3 NoIR through picamera2 (libcamera)
    IrLed      : the IR LED array, switched by one GPIO pin

Kept separate from vision.py so the vision code can be tested without a Pi.
"""
import numpy as np

from config import CAMERA, LED, CameraConfig, LedConfig


class NoirCamera:
    """Fixed-exposure, manual-focus camera that returns only the luma plane."""

    def __init__(self, cfg: CameraConfig = CAMERA) -> None:
        # Imported here so that merely importing this file works on a PC.
        from libcamera import controls
        from picamera2 import Picamera2

        self._w, self._h = cfg.size
        self._cam = Picamera2()

        # YUV420 means the camera's ISP hands us a ready-made grayscale (Y)
        # plane. That is the cheapest possible format for the Pi's CPU and, with
        # IR light, also the best signal-to-noise (all Bayer colours see IR).
        config = self._cam.create_video_configuration(
            main={"size": cfg.size, "format": "YUV420"},
            buffer_count=4,
            controls={
                "FrameDurationLimits": (int(1e6 / cfg.fps),) * 2,  # lock fps
                "AeEnable": False,               # fixed exposure (see config.py)
                "ExposureTime": cfg.exposure_us,
                "AnalogueGain": cfg.analogue_gain,
                "AwbEnable": False,
                "AfMode": controls.AfModeEnum.Manual,   # fixed focus
                "LensPosition": cfg.lens_position,
            },
        )
        self._cam.configure(config)
        self._cam.start()

    def grab_luma(self) -> np.ndarray:
        """Block until the next frame arrives, return its Y plane (H x W uint8)."""
        yuv = self._cam.capture_array("main")    # shape (H * 3/2, W)
        return yuv[: self._h, : self._w]         # first H rows are the Y plane

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class IrLed:
    """
    IR LED array on one GPIO pin (use a transistor/MOSFET: a GPIO pin cannot
    supply the current of an LED array by itself).
    """

    def __init__(self, cfg: LedConfig = LED) -> None:
        from gpiozero import DigitalOutputDevice

        self._pulse = cfg.pulse
        # Constant mode: switch on now. Pulse mode: start off, step() flips it.
        self._dev = DigitalOutputDevice(cfg.gpio_pin, initial_value=not cfg.pulse)

    def step(self) -> None:
        """Call exactly once per camera frame. Flips the LED in pulse mode."""
        if self._pulse:
            self._dev.toggle()

    def close(self) -> None:
        self._dev.off()      # leave the LEDs dark on exit
        self._dev.close()
