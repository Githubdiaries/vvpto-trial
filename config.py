"""
config.py - the ONE place where every tunable setting lives.

Copy the whole `veinscope/` folder to BOTH boards. Edit values here only; no
other file contains magic numbers.

    Raspberry Pi 4  : runs pi_main.py      (camera + IR LED + vein vision)
    BeagleBone      : runs bbb_display.py  (receives mask, drives DLP projector)

All settings are plain frozen dataclasses so a typo raises an error instead of
silently creating a new variable.
"""
from dataclasses import dataclass
from typing import Tuple


# --------------------------------------------------------------------------
# Ethernet link between the two boards (direct cable, static IPs)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NetConfig:
    pi_ip: str = "10.0.0.1"      # Raspberry Pi eth0 (set this as a static IP)
    bbb_ip: str = "10.0.0.2"     # BeagleBone eth0   (set this as a static IP)
    port: int = 5005             # BeagleBone listens here, Pi connects to it


# --------------------------------------------------------------------------
# Camera Module 3 NoIR (IMX708) on the Pi
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    # Capture resolution. Must be even numbers. 640x480 is the sweet spot for a
    # Pi 4 with 2 GB: low CPU load, still plenty of detail for veins.
    size: Tuple[int, int] = (640, 480)
    fps: int = 30

    # FIXED exposure and gain. Auto-exposure must stay OFF: the vein pipeline
    # compares consecutive frames, so brightness must only change because the
    # IR LED switched, not because the camera "helpfully" adapted.
    # Tune so that the LED-on frame is bright but NOT clipped (pixels < 250).
    exposure_us: int = 8000      # microseconds (8 ms)
    analogue_gain: float = 2.0

    # Manual focus. Value is in dioptres = 1 / distance_in_metres.
    # 3.0 -> focused at ~33 cm. 2.0 -> 50 cm. 5.0 -> 20 cm.
    lens_position: float = 3.0


# --------------------------------------------------------------------------
# IR LED array (switched through a transistor / MOSFET by a Pi GPIO pin)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LedConfig:
    gpio_pin: int = 18           # BCM numbering. Change to your wiring.

    # True  : LED flips ON/OFF every frame -> ambient light is cancelled by
    #         subtracting consecutive frames (software "bandpass", see README).
    # False : LED stays on, no ambient rejection (useful for first bring-up).
    pulse: bool = True


# --------------------------------------------------------------------------
# Vein vision pipeline (vision.py)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionConfig:
    # Internal processing size. Half of capture size = 4x less work. This is
    # the main knob for frame rate on the Pi 4 (320x240 ~ 20-30 fps expected).
    work_size: Tuple[int, int] = (320, 240)

    # Minimum change in whole-frame mean brightness between an LED-on and an
    # LED-off frame (0-255 scale). Below this the LED modulation is considered
    # lost and a BLANK mask is sent so nothing stale is projected.
    led_min_delta: float = 1.0

    # Temporal smoothing of the LED-on minus LED-off signal (0..1).
    # Lower = cleaner but laggier when the arm moves.
    signal_alpha: float = 0.5

    # ROI (skin area) = pixels brighter than this fraction of the 99.5th
    # percentile of the LED signal. Background is dark because only things the
    # LED lights up produce a signal.
    roi_threshold_frac: float = 0.25
    roi_erode_px: int = 9        # shrink ROI so the skin edge is not "a vein"
    min_roi_pixels: int = 2000   # fewer than this = no hand in view = blank

    # Flat-field: sigma of the blur that estimates the (uneven) LED lighting.
    flat_sigma: float = 15.0

    # Ridge (vein) filter scales, in pixels at work_size. Vein half-width.
    ridge_sigmas: Tuple[float, ...] = (1.5, 2.5)

    # Hysteresis thresholds as percentiles of the ridge response inside the ROI.
    # Raise both for a cleaner/sparser result, lower for more veins.
    high_percentile: float = 93.0
    low_percentile: float = 85.0
    # Absolute floor so that a hand with NO visible veins stays empty instead
    # of being filled with noise.
    min_ridge: float = 0.03
    min_blob_pixels: int = 40    # remove specks smaller than this

    # Temporal smoothing of the final mask (0..1). Prevents flicker.
    output_alpha: float = 0.5


# --------------------------------------------------------------------------
# Projector side (bbb_display.py). Command-line flags override these.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectorConfig:
    # Native resolution of YOUR DLP projector. Not known to the code author:
    # check its datasheet and edit. Ignored when drawing to a framebuffer
    # (the framebuffer's own size is used then).
    size: Tuple[int, int] = (854, 480)

    # Alignment of the camera image onto the projected image.
    flip_h: bool = False
    flip_v: bool = False
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0

    # Colour of the projected veins in BGR order. Green is easy to see on skin.
    color_bgr: Tuple[int, int, int] = (0, 255, 0)

    # If no fresh frame arrives for this long, the projector goes dark.
    stale_seconds: float = 1.0


NET = NetConfig()
CAMERA = CameraConfig()
LED = LedConfig()
VISION = VisionConfig()
PROJECTOR = ProjectorConfig()
