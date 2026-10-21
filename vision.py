"""
vision.py - turns raw camera frames into a clean "vein mask".

Pure OpenCV + NumPy, no hardware access, so it can be tested on any computer.

Data flow (every stage is cheap on purpose, the Pi 4 2GB has to keep up):

    luma frame (640x480)
      -> downscale to work_size (320x240)
      -> LED lock-in: (LED-on frame) - (LED-off frame)   <- software "bandpass"
      -> temporal average                                 <- denoise for free
      -> ROI mask (where is the skin?)
      -> flat-field (remove uneven LED lighting)
      -> CLAHE (local contrast)
      -> multi-scale Hessian ridge filter (dark lines = veins)
      -> hysteresis threshold + small-blob removal
      -> soft, temporally smoothed 8-bit mask
"""
from typing import Dict, Optional

import cv2
import numpy as np

from config import VISION, VisionConfig


class VeinPipeline:
    """Stateful pipeline: feed it frames with push(), it returns masks."""

    def __init__(self, cfg: VisionConfig = VISION, pulse: bool = True) -> None:
        self.cfg = cfg
        self.pulse = pulse                 # True when the IR LED is blinking

        self.led_ok = True                 # False when no LED blinking is seen
        self.debug: Dict[str, np.ndarray] = {}   # intermediate images (preview)

        # ---- state carried between frames --------------------------------
        self._prev: Optional[np.ndarray] = None   # previous frame (float32)
        self._prev_mean = 0.0
        self._signal: Optional[np.ndarray] = None  # smoothed LED signal
        self._out: Optional[np.ndarray] = None     # smoothed output mask

        # ---- reusable objects (creating them per frame would be slow) ----
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        k = cfg.roi_erode_px
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        w, h = cfg.work_size
        self._blank = np.zeros((h, w), np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(self, luma: np.ndarray) -> Optional[np.ndarray]:
        """
        Feed one grayscale frame (uint8, any size).

        Returns a uint8 mask of cfg.work_size (255 = vein) or None while the
        pipeline is still warming up (needs two frames).
        A BLANK mask is returned whenever the result cannot be trusted
        (LED modulation lost, no hand in view) so nothing stale is projected.
        """
        cfg = self.cfg
        small = cv2.resize(luma, cfg.work_size, interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)

        signal = self._led_lock_in(small)
        if signal is None:
            return None                              # warming up
        if not self.led_ok:
            return self._fade_to_blank()             # LED blink not detected

        # Temporal average: random sensor noise shrinks, veins stay.
        a = cfg.signal_alpha
        if self._signal is None:
            self._signal = signal
        else:
            self._signal = cv2.addWeighted(signal, a, self._signal, 1.0 - a, 0.0)

        roi = self._skin_roi(self._signal)
        if int(roi.sum()) < cfg.min_roi_pixels:
            return self._fade_to_blank()             # no hand in view

        flat = self._flat_field(self._signal, roi)
        enhanced = self._local_contrast(flat, roi)
        ridge = self._ridge_response(enhanced) * roi
        veins = self._hysteresis(ridge, roi)

        # Soften edges (nicer projection) and smooth over time (no flicker).
        soft = cv2.GaussianBlur(veins.astype(np.float32), (0, 0), 1.0)
        oa = cfg.output_alpha
        self._out = soft if self._out is None else cv2.addWeighted(
            soft, oa, self._out, 1.0 - oa, 0.0)
        mask = (np.clip(self._out, 0.0, 1.0) * 255).astype(np.uint8)

        self.debug = {
            "signal": self._to_u8(self._signal),
            "enhanced": (enhanced * 255).astype(np.uint8),
            "mask": mask,
        }
        return mask

    # ------------------------------------------------------------------
    # Stage 1: LED lock-in (software replacement for the missing bandpass)
    # ------------------------------------------------------------------
    def _led_lock_in(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Return (LED-on) minus (LED-off) so only light that blinks with the LED
        survives. Sunlight, room lights and the projector's own light are
        (nearly) the same in both frames and cancel out.

        We do not need to know which frame was "on": the brighter frame of a
        consecutive pair is the LED-on one. That makes the scheme immune to the
        camera's pipeline delay.
        """
        if not self.pulse:
            return frame                      # LED always on: no rejection

        mean = float(frame.mean())
        if self._prev is None:
            self._prev, self._prev_mean = frame, mean
            return None

        delta = mean - self._prev_mean
        on, off = (frame, self._prev) if delta >= 0 else (self._prev, frame)
        self._prev, self._prev_mean = frame, mean

        self.led_ok = abs(delta) >= self.cfg.led_min_delta
        if not self.led_ok:
            return np.zeros_like(frame)
        return np.maximum(on - off, 0.0)      # negative values are noise

    # ------------------------------------------------------------------
    # Stage 2: where is the skin?
    # ------------------------------------------------------------------
    def _skin_roi(self, signal: np.ndarray) -> np.ndarray:
        """Binary float mask (1.0 = skin). Background has ~0 LED signal."""
        top = float(np.percentile(signal, 99.5))
        if top < 1.0:
            return np.zeros_like(signal)
        roi = (signal > self.cfg.roi_threshold_frac * top).astype(np.uint8)
        roi = cv2.erode(roi, self._erode_kernel)
        return roi.astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 3: remove uneven illumination
    # ------------------------------------------------------------------
    def _flat_field(self, signal: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """
        Divide by a heavily blurred copy of itself to flatten LED hot-spots.
        "Normalised convolution" (blur the signal AND the ROI, then divide)
        stops the dark background from pulling the estimate down at the edges.
        """
        s = self.cfg.flat_sigma
        num = cv2.GaussianBlur(signal * roi, (0, 0), s)
        den = cv2.GaussianBlur(roi, (0, 0), s) + 1e-3
        flat = signal / (num / den + 1e-3)    # skin is now ~1.0, veins < 1.0
        return np.where(roi > 0, flat, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 4: local contrast
    # ------------------------------------------------------------------
    def _local_contrast(self, flat: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """CLAHE on the flattened image, returned as float32 in 0..1."""
        u8 = np.clip((flat - 0.5) * 255.0, 0, 255).astype(np.uint8)  # 0.5..1.5
        return self._clahe.apply(u8).astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Stage 5: ridge (vein) detector
    # ------------------------------------------------------------------
    def _ridge_response(self, img: np.ndarray) -> np.ndarray:
        """
        Multi-scale Hessian line detector for DARK lines on a bright surface.

        For every pixel the 2x2 Hessian (second derivatives) has two
        eigenvalues: curvature across the line (large, positive for a dark
        valley) and along it (about zero). "across minus |along|" is large for
        lines, ~0 for flat skin and ~0 for round blobs (both curvatures big).
        Multiplying by sigma^2 makes the scales comparable.
        """
        best = np.zeros_like(img)
        for s in self.cfg.ridge_sigmas:
            g = cv2.GaussianBlur(img, (0, 0), s)
            dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
            dxy = 0.25 * cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)  # Sobel gives 4x

            mean = 0.5 * (dxx + dyy)
            spread = np.sqrt(0.25 * (dxx - dyy) ** 2 + dxy ** 2)
            across = mean + spread           # bigger eigenvalue
            along = mean - spread            # smaller eigenvalue
            best = np.maximum(best, (across - np.abs(along)) * s * s)
        return np.clip(best, 0.0, None)

    # ------------------------------------------------------------------
    # Stage 6: binarise
    # ------------------------------------------------------------------
    def _hysteresis(self, ridge: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """
        Two thresholds: weak pixels only survive if connected to a strong one,
        so veins stay continuous while isolated noise is dropped.
        """
        cfg = self.cfg
        vals = ridge[roi > 0]
        hi_t = max(float(np.percentile(vals, cfg.high_percentile)), cfg.min_ridge)
        lo_t = max(float(np.percentile(vals, cfg.low_percentile)), 0.5 * cfg.min_ridge)

        strong = ridge >= hi_t
        weak = (ridge >= lo_t).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)

        keep = np.zeros(n, dtype=bool)
        keep[np.unique(labels[strong])] = True                 # touches a seed
        keep &= stats[:, cv2.CC_STAT_AREA] >= cfg.min_blob_pixels
        keep[0] = False                                        # label 0 = background
        return keep[labels]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fade_to_blank(self) -> np.ndarray:
        """Used when the result is untrustworthy: reset state, project nothing."""
        self._signal = None
        self._out = None
        return self._blank

    @staticmethod
    def _to_u8(img: np.ndarray) -> np.ndarray:
        top = float(img.max()) + 1e-6
        return np.clip(img * (255.0 / top), 0, 255).astype(np.uint8)
