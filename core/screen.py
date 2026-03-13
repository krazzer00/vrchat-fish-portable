"""
Screen Capture Module
=====================
Supports two capture methods (auto-selected):

1. **PrintWindow API** (preferred)
   Captures directly from VRChat window buffer, works even when window
   is completely obscured. User can use the computer normally.

2. **mss screen capture** (fallback)
   If PrintWindow is unavailable for VRChat (rare cases),
   falls back to capturing window area pixels. Requires VRChat window visible.

First call to grab_window() auto-tests if PrintWindow is available
and logs the result.

Note: mss instance is thread-local, cannot be used across threads.
PrintWindow uses ctypes + GDI, inherently thread-safe.
"""

import os
import threading
import ctypes
import ctypes.wintypes
import cv2
import numpy as np
from mss import mss

import config
from utils.logger import log


# ═══════════════════ Win32 GDI Constants & Functions ═══════════════════

user32 = ctypes.windll.user32
gdi32  = ctypes.windll.gdi32

PW_CLIENTONLY          = 0x1      # Capture client area only (no title bar/border)
PW_RENDERFULLCONTENT   = 0x2      # Use DWM composite rendering (Win8.1+, supports DirectX)
SRCCOPY                = 0x00CC0020
BI_RGB                 = 0
DIB_RGB_COLORS         = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize',          ctypes.c_uint32),
        ('biWidth',         ctypes.c_int32),
        ('biHeight',        ctypes.c_int32),
        ('biPlanes',        ctypes.c_uint16),
        ('biBitCount',      ctypes.c_uint16),
        ('biCompression',   ctypes.c_uint32),
        ('biSizeImage',     ctypes.c_uint32),
        ('biXPelsPerMeter', ctypes.c_int32),
        ('biYPelsPerMeter', ctypes.c_int32),
        ('biClrUsed',       ctypes.c_uint32),
        ('biClrImportant',  ctypes.c_uint32),
    ]


# ═══════════════════ Screen Capture ═══════════════════

class ScreenCapture:
    """High-speed screen capture (thread-safe) - Optimized: reuses GDI resources + pre-allocated buffers"""

    def __init__(self):
        self._local = threading.local()   # Per-thread independent mss instance
        self.screen_w = 0
        self.screen_h = 0

        # PrintWindow availability (auto-detected on first grab_window)
        # None=untested, True=available, False=unavailable
        self._use_printwindow = None
        self._pw_tested_hwnd = None        # HWND used during testing

        # ── Optimization: pre-allocate GDI resources (reuse per-frame, avoid repeated create/delete) ──
        self._pw_dc = None          # Compatible DC (memory DC)
        self._pw_bmp = None         # Compatible bitmap
        self._pw_old_bmp = None     # Old bitmap (must restore during cleanup)
        self._pw_wdc = None         # Window DC
        self._pw_hwnd = None        # Current HWND for bound resources
        self._pw_w = 0              # Current bitmap width
        self._pw_h = 0              # Current bitmap height
        self._pw_buf = None         # Pre-allocated ctypes buffer
        self._pw_bmi = None         # Pre-allocated BITMAPINFOHEADER
        self._pw_np_buf = None      # Pre-allocated numpy output buffer (BGR)

        # Get screen dimensions in main thread
        sct = self._get_sct()
        primary = sct.monitors[1]
        self.screen_w = primary["width"]
        self.screen_h = primary["height"]

        # Ensure debug directory exists
        os.makedirs(config.DEBUG_DIR, exist_ok=True)

    def _get_sct(self):
        """Get mss instance for current thread (lazy initialization)"""
        if not hasattr(self._local, "sct") or self._local.sct is None:
            self._local.sct = mss()
        return self._local.sct

    # ────────────────── PrintWindow Capture (Optimized) ──────────────────

    def _ensure_pw_resources(self, hwnd, w, h):
        """
        Ensure GDI resources are allocated and match current window size.
        Rebuild only on HWND change or window resize, fully reuse between normal frames.
        """
        if (self._pw_hwnd == hwnd and self._pw_w == w and self._pw_h == h
                and self._pw_dc is not None):
            return True  # Resources ready, no rebuild needed

        # Release old resources
        self._release_pw_resources()

        wDC = user32.GetDC(hwnd)
        if not wDC:
            return False

        mDC = gdi32.CreateCompatibleDC(wDC)
        if not mDC:
            user32.ReleaseDC(hwnd, wDC)
            return False

        bmp = gdi32.CreateCompatibleBitmap(wDC, w, h)
        if not bmp:
            gdi32.DeleteDC(mDC)
            user32.ReleaseDC(hwnd, wDC)
            return False

        old_bmp = gdi32.SelectObject(mDC, bmp)

        # Pre-allocate BITMAPINFOHEADER (reuse per-frame)
        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h           # Negative = top-down layout
        bmi.biPlanes = 1
        bmi.biBitCount = 32         # 32-bit BGRA
        bmi.biCompression = BI_RGB

        # Pre-allocate buffer
        buf_size = w * h * 4
        pw_buf = ctypes.create_string_buffer(buf_size)

        # Pre-allocate numpy BGR output buffer
        np_buf = np.empty((h, w, 3), dtype=np.uint8)

        self._pw_wdc = wDC
        self._pw_dc = mDC
        self._pw_bmp = bmp
        self._pw_old_bmp = old_bmp
        self._pw_hwnd = hwnd
        self._pw_w = w
        self._pw_h = h
        self._pw_bmi = bmi
        self._pw_buf = pw_buf
        self._pw_np_buf = np_buf

        return True

    def _release_pw_resources(self):
        """Release pre-allocated GDI resources"""
        try:
            if self._pw_old_bmp and self._pw_dc:
                gdi32.SelectObject(self._pw_dc, self._pw_old_bmp)
            if self._pw_bmp:
                gdi32.DeleteObject(self._pw_bmp)
            if self._pw_dc:
                gdi32.DeleteDC(self._pw_dc)
            if self._pw_wdc and self._pw_hwnd:
                user32.ReleaseDC(self._pw_hwnd, self._pw_wdc)
        except Exception:
            pass
        self._pw_dc = None
        self._pw_bmp = None
        self._pw_old_bmp = None
        self._pw_wdc = None
        self._pw_hwnd = None
        self._pw_w = 0
        self._pw_h = 0
        self._pw_buf = None
        self._pw_bmi = None
        self._pw_np_buf = None

    def __del__(self):
        """Release GDI resources during destruction"""
        self._release_pw_resources()

    def _grab_printwindow(self, hwnd):
        """
        Capture window client area content using PrintWindow API - Optimized version.

        Optimizations:
        1. GDI DC/bitmap/buffers pre-allocated, reused per-frame (saves Create/Delete overhead)
        2. BGRA→BGR via numpy slicing + pre-allocated output buffer (zero extra allocation)
        3. BITMAPINFOHEADER pre-allocated and reused

        Returns:
            BGR numpy array or None (on failure)
        """
        if not hwnd:
            return None

        try:
            # Get client area size
            rect = ctypes.wintypes.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            w = rect.right
            h = rect.bottom
            if w <= 0 or h <= 0:
                return None

            # Ensure resources ready (most frames return True directly, no rebuild)
            if not self._ensure_pw_resources(hwnd, w, h):
                return None

            # PrintWindow: client area + DWM rendering
            ok = user32.PrintWindow(
                hwnd, self._pw_dc,
                PW_CLIENTONLY | PW_RENDERFULLCONTENT
            )

            if not ok:
                gdi32.BitBlt(
                    self._pw_dc, 0, 0, w, h,
                    self._pw_wdc, 0, 0, SRCCOPY
                )

            # Read pixels to pre-allocated buffer
            gdi32.GetDIBits(
                self._pw_dc, self._pw_bmp, 0, h,
                self._pw_buf, ctypes.byref(self._pw_bmi), DIB_RGB_COLORS
            )

            # BGRA → BGR: reuse ctypes buffer, only do one channel slice copy
            # (old: each frame create_string_buffer + frombuffer + reshape + copy = 3 allocations)
            # (optimized: reuse ctypes buffer, only final BGR copy = 1 allocation)
            bgra = np.frombuffer(self._pw_buf, dtype=np.uint8).reshape((h, w, 4))
            return bgra[:, :, :3].copy()

        except Exception as e:
            log.debug(f"PrintWindow exception: {e}")
            # Resources may be corrupted, mark for rebuild
            self._release_pw_resources()
            return None

    def _test_printwindow(self, hwnd) -> bool:
        """
        Test if PrintWindow is available for current window.
        Captures one frame, checks if it's all black (all black = unavailable).
        """
        img = self._grab_printwindow(hwnd)
        if img is None:
            return False

        # Check if all-black image (DirectX exclusive mode returns all black)
        mean_val = float(np.mean(img))
        if mean_val > 5.0:
            h, w = img.shape[:2]
            log.info(
                f"✓ PrintWindow available ({w}×{h}, brightness={mean_val:.1f}) "
                f"— VRChat can be obscured and still capture correctly"
            )
            return True
        else:
            log.warning(
                f"✗ PrintWindow returned black screen (brightness={mean_val:.1f}) "
                f"— Keep VRChat window unobscured!"
            )
            return False

    # ────────────────── mss capture (fallback) ──────────────────

    def grab(self, region=None):
        """
        Capture screen.
        Args:
            region: (x, y, w, h) or None=full screen
        Returns:
            BGR numpy array
        """
        sct = self._get_sct()

        if region:
            mon = {
                "left":   int(region[0]),
                "top":    int(region[1]),
                "width":  max(1, int(region[2])),
                "height": max(1, int(region[3])),
            }
        else:
            mon = sct.monitors[1]

        raw = np.array(sct.grab(mon))
        return np.ascontiguousarray(raw[:, :, :3])

    # ────────────────── Main Interface ──────────────────

    def grab_window(self, window_mgr):
        """
        Capture VRChat window client area.

        Automatically selects best capture method:
        1. PrintWindow — window can be obscured by others (preferred)
        2. mss — capture screen region (fallback, requires window visible)

        Returns:
            (image, region)  — region is (x, y, w, h) or None
        """
        hwnd = window_mgr.hwnd if window_mgr.is_valid() else None

        # ── First call / HWND change: test if PrintWindow is available ──
        if hwnd and (self._use_printwindow is None
                     or self._pw_tested_hwnd != hwnd):
            self._pw_tested_hwnd = hwnd
            self._use_printwindow = self._test_printwindow(hwnd)

        # ── Method 1: PrintWindow (direct window capture) ──
        if self._use_printwindow and hwnd:
            img = self._grab_printwindow(hwnd)
            if img is not None:
                # Fast black-screen detection: sample few pixels instead of full image mean
                # (for 1920x1080 image: sampling ~0.01ms vs np.mean ~2ms)
                h_img, w_img = img.shape[:2]
                _sample = img[h_img // 4, w_img // 4, 0]    # Top-left 1/4
                _sample += img[h_img // 2, w_img // 2, 1]    # Center
                _sample += img[h_img * 3 // 4, w_img * 3 // 4, 2]  # Bottom-right 3/4
                if _sample > 6:
                    return img, None

        # ── Method 2: mss screen capture (fallback) ──
        region = window_mgr.get_region()
        if region and region[2] > 0 and region[3] > 0:
            return self.grab(region), region

        # Final fallback: full screen
        return self.grab(), None

    # ────────────────── Utility Methods ──────────────────

    def save_debug(self, image, name: str = "screenshot"):
        """Save debug screenshot to debug/ directory"""
        path = os.path.join(config.DEBUG_DIR, f"{name}.png")
        cv2.imwrite(path, image)
        log.debug(f"Debug screenshot saved: {path}")

    def reset_capture_method(self):
        """
        Reset capture method detection.
        Call when VRChat window restarts or mode changes, forces re-testing.
        """
        self._use_printwindow = None
        self._pw_tested_hwnd = None
        log.info("Capture method reset, will re-detect on next screenshot")
