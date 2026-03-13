"""
Fishing bot main logic
================
State machine: IDLE → CASTING → WAITING → HOOKING → FISHING → (loop)

Designed to run in background thread, communicates with GUI via shared properties.
"""

import time
import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import config
from core.window import WindowManager
from core.screen import ScreenCapture
from core.detector import ImageDetector
from core.input_ctrl import InputController
from utils.logger import log

import ctypes
import csv
from collections import deque
import numpy as np

_yolo_detector = None
_yolo_device_used = None

def _get_yolo_detector(force_reload=False):
    """Lazy-load YOLO detector (avoid errors if ultralytics not installed)"""
    global _yolo_detector, _yolo_device_used
    if force_reload:
        _yolo_detector = None
    if _yolo_detector is None or _yolo_device_used != config.YOLO_DEVICE:
        from core.yolo_detector import YoloDetector
        _yolo_detector = YoloDetector(config.YOLO_MODEL, conf=config.YOLO_CONF)
        _yolo_device_used = config.YOLO_DEVICE
    return _yolo_detector


class FishingBot:
    """VRChat automatic fishing bot"""

    # Fish template → English name + debug box color (BGR)
    FISH_DISPLAY = {
        "fish_black":   ("Black Fish",  (80, 80, 80)),
        "fish_white":   ("White Fish",  (255, 255, 255)),
        "fish_copper":  ("Copper Fish",  (50, 127, 180)),
        "fish_green":   ("Green Fish",  (0, 255, 0)),
        "fish_blue":    ("Blue Fish",  (255, 150, 0)),
        "fish_purple":  ("Purple Fish",  (200, 50, 200)),
        "fish_golden":  ("Gold Fish",  (0, 215, 255)),
        "fish_pink":    ("Pink Fish",  (180, 105, 255)),
        "fish_red":     ("Red Fish",  (0, 0, 255)),
        "fish_rainbow": ("Rainbow Fish",  (0, 255, 255)),
    }

    def __init__(self):
        self.window   = WindowManager(config.WINDOW_TITLE)
        self.screen   = ScreenCapture()
        self.detector = ImageDetector(config.IMG_DIR, config.TEMPLATE_FILES)
        self.input    = InputController(self.window)

        self.yolo = None
        if config.USE_YOLO:
            try:
                self.yolo = _get_yolo_detector()
            except Exception as e:
                log.warning(f"[YOLO] Initialization failed: {e}")

        # ── Shared state (GUI reads) ──
        self.running    = False
        self.debug_mode = False
        self.fish_count = 0
        self.state      = "Ready"

        # ── PD controller state ──
        self._bar_prev_cy   = None       # Previous frame bar center Y
        self._bar_prev_time = None       # Previous frame timestamp
        self._bar_velocity  = 0.0        # Bar velocity estimate (px/s, positive=down, negative=up)
        self._last_hold     = None       # Last hold duration (fallback)
        self._last_fish_cy  = None       # Last fish center Y (fallback)
        self._fish_prev_cy  = None       # Previous frame fish center Y (calculate fish speed)
        self._fish_prev_time = None      # Previous frame fish timestamp

        # ── Debug overlay (independent thread, doesn't block fishing logic) ──
        self._last_overlay_time = 0
        self._fps = 0.0
        self._frame_times = deque(maxlen=30)  # Optimize: deque instead of list
        self._debug_frame = None         # Latest frame to display
        self._debug_lock = threading.Lock()
        self._debug_thread = None
        self._debug_stop_event = threading.Event()  # Stop debug thread safely

        # ── Rotation compensation state ──
        self._track_angle   = 0.0        # Track tilt angle (degrees)
        self._need_rotation = False      # Whether rotation compensation needed

        # ── Fish/bar position smoothing (reduce detection jitter) ──
        self._fish_smooth_cy = None      # Smoothed fish center Y
        self._current_fish_name = ""     # Current detected fish template (e.g. "fish_blue")
        self._bar_locked_cx  = None      # Track X-axis lock (shared by bar and fish)
        self._pool = ThreadPoolExecutor(max_workers=2)

        # ── Imitation learning ──
        self._il_history = deque(maxlen=config.IL_HISTORY_LEN)
        self._il_writer = None       # CSV writer (record mode)
        self._il_file = None         # CSV file handle
        self._il_prev_fish_cy = None # Previous frame fish Y (calculate fish displacement)
        self._il_mouse_prev = 0      # Previous frame mouse state
        self._il_log_counter = 0     # Log throttle counter
        self._il_policy = None       # Trained model
        self._il_device = "cpu"
        self._il_norm_mean = None    # Feature normalization mean
        self._il_norm_std = None     # Feature normalization std dev
        if config.IL_USE_MODEL:
            self._load_il_policy()

    # ══════════════════════════════════════════════════════
    #  Sleep with debug overlay refresh
    # ══════════════════════════════════════════════════════

    def _sleep_with_overlay(self, duration, status_text=""):
        """
        Sleep for `duration` seconds while continuously refreshing the debug overlay.
        This prevents the debug window from freezing during long waits (cast, hook, etc).
        """
        t0 = time.time()
        while time.time() - t0 < duration:
            if not self.running:
                break
            try:
                screen = self._grab()
                self._show_debug_overlay(screen, status_text=status_text)
            except Exception:
                pass
            remaining = duration - (time.time() - t0)
            time.sleep(min(0.033, max(0.001, remaining)))  # ~30fps cap

    # ══════════════════════════════════════════════════════
    #  Screen capture
    # ══════════════════════════════════════════════════════

    # Optimize: pre-allocate blank frame (avoid recreating on failure)
    _EMPTY_FRAME = np.zeros((100, 100, 3), dtype=np.uint8)

    def _grab(self):
        """Capture VRChat window client area, ensure non-empty BGR image returned"""
        try:
            img, _ = self.screen.grab_window(self.window)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass
        return self._EMPTY_FRAME

    def _grab_rotated(self):
        """Capture window, rotate if track is tilted to make it vertical"""
        img = self._grab()
        if self._need_rotation:
            return self._rotate_for_detection(img)
        return img

    def _rotate_for_detection(self, screen):
        """
        Rotate image to make tilted fishing track vertical.

        Logic: track tilted θ° → rotate image -θ° → track becomes vertical
        After rotation, all existing template matching code works normally.
        """
        h, w = screen.shape[:2]
        center = (w / 2.0, h / 2.0)

        # getRotationMatrix2D: positive angle is clockwise rotation in image coords
        # track tilted right θ° → need counterclockwise rotation θ° → pass -θ
        M = cv2.getRotationMatrix2D(center, -self._track_angle, 1.0)

        # Expand canvas to avoid clipping rotated content
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        return cv2.warpAffine(
            screen, M, (new_w, new_h), borderValue=(0, 0, 0)
        )

    # ══════════════════════════════════════════════════════
    #  Step 1: Cast rod
    # ══════════════════════════════════════════════════════

    def _cast_rod(self):
        self.state = "Casting"
        if config.IL_RECORD:
            log.info("[Cast] Record mode — cast manually (click mouse)")
        else:
            log.info("[Cast] Shaking head -> casting...")
            self.input.shake_head()
            self._sleep_with_overlay(0.15, status_text="Casting...")
            self.input.click()
        self._sleep_with_overlay(config.CAST_DELAY, status_text="Casting... waiting")

    # ══════════════════════════════════════════════════════
    #  Step 2: Wait for bite
    # ══════════════════════════════════════════════════════

    def _wait_for_bite(self) -> bool:
        self.state = "Waiting for bite"
        if config.IL_RECORD:
            wait_s = config.MINIGAME_TIMEOUT
            log.info(f"[Wait] Record mode — operate manually, waiting for minigame (max {wait_s:.0f}s)...")
        else:
            wait_s = config.BITE_FORCE_HOOK
            log.info(f"[Wait] Waiting {wait_s:.0f}s before auto-hook...")

        t0 = time.time()
        while self.running:
            elapsed = time.time() - t0
            if elapsed >= wait_s:
                log.info(f"[Hook] Waited {elapsed:.1f}s, auto-hooking")
                return True

            try:
                screen = self._grab()
                self._show_debug_overlay(
                    screen,
                    status_text=f"Waiting for hook ({elapsed:.0f}/{wait_s:.0f}s)"
                )
            except Exception:
                pass

            time.sleep(0.05)

        return False

    # ══════════════════════════════════════════════════════
    #  Step 3: Hook fish
    # ══════════════════════════════════════════════════════

    def _hook_fish(self):
        self.state = "Hooking"
        if config.IL_RECORD:
            log.info("[Hook] Record mode — hook manually (click mouse)")
        else:
            log.info("[Hook] Clicking to hook!")
            self._sleep_with_overlay(config.HOOK_PRE_DELAY, "Hooking...")
            self.input.click()
        self._sleep_with_overlay(config.HOOK_POST_DELAY, "Hooked! Waiting for minigame UI...")

    def _verify_minigame(self) -> bool:
        """
        After hook, verify that the fishing minigame UI actually appeared.

        Confirm when we detect UI in N total frames (not strictly consecutive).
        Prefer YOLO detection if available, fallback to template matching.
        """
        self.state = "Verifying minigame"
        log.info("[🔍 Verifying] Quick UI detection...")

        t0 = time.time()
        hit_count = 0
        required = config.VERIFY_CONSECUTIVE
        _use_yolo = config.USE_YOLO and self.yolo is not None

        # Reset rotation state
        self._track_angle = 0.0
        self._need_rotation = False
        detected_angle = None

        while self.running and (time.time() - t0 < config.VERIFY_TIMEOUT):
            screen = self._grab()
            found = False

            self._show_debug_overlay(
                screen,
                status_text=f"🔍 Verifying UI ({hit_count}/{required})"
            )

            _roi = config.DETECT_ROI

            # ── Prefer YOLO detection ──
            if _use_yolo:
                try:
                    det = self.yolo.detect(screen, _roi)
                    if det.get("bar") and det.get("track"):
                        yb = det["bar"]
                        yt = det["track"]
                        bar_cx = yb[0] + yb[2] // 2
                        track_cx = yt[0] + yt[2] // 2
                        if abs(bar_cx - track_cx) < 150:
                            found = True
                            detected_angle = 0.0
                except Exception:
                    pass

            # ── Template matching fallback ──
            if not found:
                bar = self.detector.find_multiscale(
                    screen, "bar", config.THRESH_BAR,
                    scales=config.BAR_SCALES,
                    search_region=_roi,
                )
                track = self.detector.find_multiscale(
                    screen, "track", config.THRESH_TRACK,
                    search_region=_roi,
                )

                bar_cx = (bar[0] + bar[2] // 2) if bar else None
                track_cx = (track[0] + track[2] // 2) if track else None

                if bar_cx is not None and track_cx is not None:
                    if abs(bar_cx - track_cx) < 150:
                        found = True
                        detected_angle = 0.0

            if found:
                hit_count += 1
                if hit_count >= required:
                    if detected_angle is not None:
                        self._track_angle = detected_angle
                        angle_abs = abs(self._track_angle)
                        self._need_rotation = (
                            angle_abs > config.TRACK_MIN_ANGLE
                            and angle_abs <= config.TRACK_MAX_ANGLE
                        )
                    log.info(
                        f"[✓ Confirmed] UI detected! "
                        f"(elapsed {time.time()-t0:.1f}s"
                        f", angle={self._track_angle:.1f}°)"
                    )
                    return True

            time.sleep(0.03)

        log.warning(
            f"[✗ False trigger] Minigame UI not confirmed within {config.VERIFY_TIMEOUT:.1f}s "
            f"(total hits: {hit_count}/{required}), recasting"
        )
        return False

    def _wait_for_minigame_ui(self) -> bool:
        """
        Record mode only: continuously wait for minigame UI.
        Require bar and track detected simultaneously, 3 consecutive frame confirmation.
        """
        consecutive = 0
        required = 3
        _roi = config.DETECT_ROI
        logged = False

        while self.running:
            screen = self._grab()
            self._show_debug_overlay(
                screen,
                status_text=f"[IL] Waiting for minigame... ({consecutive}/{required})"
            )

            bar = self.detector.find_multiscale(
                screen, "bar", config.THRESH_BAR,
                scales=config.BAR_SCALES, search_region=_roi,
            )
            track = self.detector.find_multiscale(
                screen, "track", config.THRESH_TRACK,
                search_region=_roi,
            )

            if bar is not None and track is not None:
                bar_cx = bar[0] + bar[2] // 2
                track_cx = track[0] + track[2] // 2
                if abs(bar_cx - track_cx) < 150:
                    consecutive += 1
                    if not logged and consecutive >= 1:
                        log.info(f"[IL] UI elements detected ({consecutive}/{required})...")
                        logged = True
                    if consecutive >= required:
                        log.info(
                            f"[IL] Minigame confirmed! ({required} consecutive frames: bar+track)"
                        )
                        return True
                else:
                    consecutive = 0
                    logged = False
            else:
                consecutive = 0
                logged = False

            time.sleep(0.1)

        return False

    # ══════════════════════════════════════════════════════
    #  Step 4: Fishing minigame
    # ══════════════════════════════════════════════════════

    def _fishing_minigame(self) -> bool:
        self.state = "Fishing minigame"
        log.info("[🐟 Fishing] Minigame started")

        # ── Imitation learning: reset state each round ──
        self._il_history.clear()
        self._il_prev_fish_cy = None
        self._il_mouse_prev = 0
        self._il_press_streak = 0
        self._il_prev_velocity = 0.0
        self._il_log_counter = 0

        if config.IL_RECORD:
            self._il_start_recording()
            log.info("[IL] Record mode: manually control bar with mouse!")
        elif config.IL_USE_MODEL:
            if self._il_policy is None:
                self._load_il_policy()
            if self._il_policy is not None:
                log.info("[IL] ★ Using imitation learning model this round ★")
            else:
                log.warning("[IL] Model load failed, fallback to PD controller")
        else:
            log.info("[PD] Using PD controller this round")

        # ★ YOLO mode (lazy-load on first use)
        if config.USE_YOLO and self.yolo is None:
            try:
                self.yolo = _get_yolo_detector()
            except Exception as e:
                log.warning(f"[YOLO] Load failed: {e}, fallback to template matching")
        _use_yolo = config.USE_YOLO and self.yolo is not None
        if _use_yolo:
            log.info("[YOLO] Using YOLO object detection")

        # ★ Enable debug report for first few seconds (troubleshoot detection)
        self.detector.debug_report = True

        # ★ PostMessage mode doesn't need window focus, only update click coords
        self.input.move_to_game_center()

        no_detect = 0
        fish_lost = 0          # ★ Consecutive frames fish missing
        frame = 0
        hold_count = 0         # Presses count
        success = False
        _skip_fish = False     # ★ Whitelist skip flag: non-target fish -> give up
        _fish_id_saved = False # ★ Fish ID screenshot save only once
        self._progress_debug_saved = False  # ★ Progress bar screenshot save once
        minigame_start = time.time()   # ★ Timer: force end on timeout
        ui_gone_count = 0              # ★ UI missing counter
        had_good_detection = False     # ★ Ever detected fish+bar successfully
        track_alive = True             # ★ Track alive (update periodically)
        obj_gone_count = 0             # ★ Consecutive insufficient object frames
        fish_gone_since = None         # ★ Fish missing start time
        bar_gone_since  = None         # ★ Bar missing start time

        # ── Reset PD controller ──
        self._bar_prev_cy   = None
        self._bar_prev_time = None
        self._bar_velocity  = 0.0
        self._last_hold     = None
        self._last_fish_cy  = None
        self._fish_prev_cy  = None
        self._fish_prev_time = None
        self._fish_smooth_cy = None
        self._bar_locked_cx  = None

        # ── Template lock variables (speed up frame detection) ──
        locked_fish_key = None       # e.g. "fish_blue"
        locked_fish_scales = None    # e.g. [0.4, 0.5, 0.6]
        locked_bar_scales = None     # e.g. [0.4, 0.5, 0.6]
        _BAR_X_HALF = config.REGION_X
        _FISH_X_HALF = max(config.REGION_X * 2, 80)

        # Initialize search region
        screen_orig = self._grab()

        # ★ Always save minigame start frame (original, unrotated)
        self.screen.save_debug(screen_orig, "minigame_start")
        h_orig, w_orig = screen_orig.shape[:2]
        log.info(f"  Screenshot size: {w_orig}×{h_orig}")

        # ★ Also refresh debug window during init
        self._show_debug_overlay(
            screen_orig, status_text="🐟 Minigame initializing..."
        )

        if self._need_rotation:
            log.info(
                f"  ► Track tilted {self._track_angle:.1f}°, "
                f"rotation compensation enabled (rotate {-self._track_angle:.1f}°)"
            )
            screen = self._rotate_for_detection(screen_orig)
        else:
            screen = screen_orig

        h_scr, w_scr = screen.shape[:2]

        if _use_yolo:
            search_region = None
            bar_search_region = None
            _regions_locked = True
            if config.DETECT_ROI:
                log.info(
                    f"  [YOLO] Using ROI: "
                    f"X={config.DETECT_ROI[0]} Y={config.DETECT_ROI[1]} "
                    f"{config.DETECT_ROI[2]}x{config.DETECT_ROI[3]}"
                )
            else:
                log.info("  [YOLO] Fullscreen detection")
        else:
            search_region, track_cx, bar_search_region = \
                self._init_search_region(screen)
            _regions_locked = False

            if track_cx is not None:
                self._bar_locked_cx = track_cx
                log.info(f"  ★ Track X-axis pre-locked: X={track_cx}")

            if search_region:
                srx, sry, srw, srh = search_region
                log.info(
                    f"  Initial fish search: X={srx}~{srx+srw} Y={sry}~{sry+srh}"
                )
            if bar_search_region:
                bsx, bsy, bsw, bsh = bar_search_region
                log.info(
                    f"  Initial bar search: X={bsx}~{bsx+bsw} "
                    f"Y={bsy}~{bsy+bsh} (lower half)"
                )

        # ★ Initial press: bar drops quickly from middle, two presses restore inertia
        if config.IL_RECORD:
            log.info("  ► Record mode - skip initial press, control manually")
        else:
            press_t = getattr(config, 'INITIAL_PRESS_TIME', 0.2)
            log.info(f"  ► Initial delay 0.5s + press {press_t}s")
            time.sleep(0.5)
            self.input.mouse_down()
            time.sleep(press_t)
            self.input.mouse_up()

        _last_progress_sr = None
        _last_track_w = None
        _last_green = 0.0
        _PROGRESS_SKIP_FRAMES = 20
        _prev_green = 0.0
        try:
            while self.running:
                frame += 1
                # ★ FPS calculation (optimize: deque auto-manages length)
                now_t = time.time()
                self._frame_times.append(now_t)
                if len(self._frame_times) >= 2:
                    dt = self._frame_times[-1] - self._frame_times[0]
                    if dt > 0:
                        self._fps = (len(self._frame_times) - 1) / dt

                screen_raw = self._grab()
                screen = self._rotate_for_detection(screen_raw) \
                    if self._need_rotation else screen_raw

                # ════════════ Timeout detection ════════════
                elapsed = time.time() - minigame_start
                if elapsed > config.MINIGAME_TIMEOUT:
                    log.info(
                        f"[⏱ Timeout] Minigame running {elapsed:.0f}s, "
                        f"exceeded {config.MINIGAME_TIMEOUT:.0f}s limit, force end"
                    )
                    break

                # ════════════ Periodically check if UI still exists ════════════
                if frame % config.UI_CHECK_FRAMES == 0 and frame > 10:
                    if _use_yolo:
                        _tc = self.yolo.detect(screen, config.DETECT_ROI)
                        track_check = _tc["track"]
                    else:
                        track_check = self.detector.find_multiscale(
                            screen, "track", 0.50
                        )
                    if track_check is None:
                        ui_gone_count += 1
                        track_alive = False
                        log.info(
                            f"[⚠ UI check] Track not detected "
                            f"({ui_gone_count}/{config.UI_GONE_LIMIT})"
                        )
                        if ui_gone_count >= config.UI_GONE_LIMIT:
                            log.info("[📋 End] Minigame UI disappeared, game ended!")
                            break
                    else:
                        ui_gone_count = 0
                        track_alive = True

                # ★ Every 60 frames ensure cursor in game window
                if frame % 60 == 0:
                    self.input.ensure_cursor_in_game()

                # ════════════ ★ Skip expensive full search when consecutive losses ════════════
                if no_detect > 3 and not _use_yolo:
                    bar_quick = self.detector.find_multiscale(
                        screen, "bar", config.THRESH_BAR,
                        bar_search_region,
                        scales=locked_bar_scales or config.BAR_SCALES,
                    )
                    if bar_quick is not None:
                        # UI may have recovered, reset count for next full detection
                        log.info(f"[✓ Recovered] Bar re-detected after {no_detect} frames missing")
                        no_detect = 0
                    else:
                        no_detect += 1
                        if no_detect > 5:
                            self.input.mouse_up()
                        if no_detect > config.TRACK_LOST_LIMIT:
                            log.info(
                                f"[📋 End] {no_detect} consecutive frames "
                                f"no valid UI detected, game ended"
                            )
                            break
                        # ★ debug window still refresh
                        self._show_debug_overlay(
                            screen_raw,
                            status_text=f"⚠ Missing {no_detect}/{config.TRACK_LOST_LIMIT}"
                        )
                        time.sleep(config.GAME_LOOP_INTERVAL)
                        continue

                # ════════════ Detect fish + bar ════════════
                fish = None
                bar = None
                fish_detect_name = ""
                _matched_key = None
                _bar_scale = 1.0

                _yolo_progress = None
                if _use_yolo:
                    # ──── YOLO: single inference detects all ────
                    _yolo_roi = config.DETECT_ROI
                    _ydet = self.yolo.detect(screen, roi=_yolo_roi)
                    fish = _ydet["fish"]
                    bar = _ydet["bar"]
                    _yolo_progress = _ydet.get("progress")
                    if fish is not None:
                        _save = not _fish_id_saved
                        _color_key = self.detector.identify_fish_type(
                            screen, fish, debug_save=_save)
                        if _save:
                            _fish_id_saved = True
                        _matched_key = _color_key
                        fish_detect_name = _color_key
                    else:
                        _matched_key = None
                        fish_detect_name = ""

                    # YOLO data collection: save full window (no ROI crop)
                    if config.YOLO_COLLECT and frame % 10 == 0:
                        _cdir = os.path.join(
                            config.BASE_DIR, "yolo", "dataset",
                            "images", "unlabeled")
                        os.makedirs(_cdir, exist_ok=True)
                        _ts = time.strftime("%Y%m%d_%H%M%S")
                        _ms = int((time.time() % 1) * 1000)
                        cv2.imwrite(
                            os.path.join(_cdir, f"{_ts}_{_ms:03d}.png"),
                            screen)

                else:
                    # ──── Template matching: original logic ────
                    _fish_sr = search_region
                    if search_region:
                        _sr_x, _sr_y, _sr_w, _sr_h = search_region
                        _new_x, _new_w = _sr_x, _sr_w
                        _new_y, _new_h = _sr_y, _sr_h
                        if self._bar_locked_cx is not None:
                            _nx = max(_sr_x,
                                      self._bar_locked_cx - _FISH_X_HALF)
                            _nx2 = min(_sr_x + _sr_w,
                                       self._bar_locked_cx + _FISH_X_HALF)
                            if _nx2 - _nx > 10:
                                _new_x, _new_w = _nx, _nx2 - _nx
                        if self._fish_smooth_cy is not None:
                            _ny = max(_sr_y,
                                      int(self._fish_smooth_cy) - 150)
                            _ny2 = min(_sr_y + _sr_h,
                                       int(self._fish_smooth_cy) + 150)
                            if _ny2 - _ny > 30:
                                _new_y, _new_h = _ny, _ny2 - _ny
                        _fish_sr = (_new_x, _new_y, _new_w, _new_h)

                    _fg, _fox, _foy = self.detector.prepare_gray(
                        screen, _fish_sr, upload_gpu=True
                    )
                    _bg, _box, _boy = self.detector.prepare_gray(
                        screen, bar_search_region, upload_gpu=True
                    )

                    _has_cuda = self.detector._use_cuda

                    def _detect_fish():
                        if locked_fish_key:
                            r = self.detector.find_multiscale(
                                screen, locked_fish_key, config.THRESH_FISH,
                                _fish_sr, scales=locked_fish_scales,
                                pre_gray=_fg, pre_offset=(_fox, _foy),
                            )
                            if r is None and _fish_sr is not search_region:
                                r = self.detector.find_multiscale(
                                    screen, locked_fish_key,
                                    config.THRESH_FISH,
                                    search_region, scales=locked_fish_scales
                                )
                            return r, locked_fish_key if r else None
                        else:
                            if _has_cuda:
                                r = self.detector.find_fish(
                                    screen, config.THRESH_FISH, _fish_sr,
                                    pre_gray=_fg, pre_offset=(_fox, _foy),
                                )
                            else:
                                _n = len(config.FISH_KEYS)
                                _grp_size = 2
                                _grp_count = ((_n + _grp_size - 1)
                                              // _grp_size)
                                _grp_idx = frame % _grp_count
                                _start = _grp_idx * _grp_size
                                _keys = config.FISH_KEYS[
                                    _start:_start + _grp_size]
                                r = self.detector.find_fish(
                                    screen, config.THRESH_FISH, _fish_sr,
                                    pre_gray=_fg, pre_offset=(_fox, _foy),
                                    keys=_keys,
                                )
                            return (r, self.detector._last_best_key
                                    if r else None)

                    def _detect_bar():
                        _scales = locked_bar_scales or config.BAR_SCALES
                        r = self.detector.find_multiscale(
                            screen, "bar", config.THRESH_BAR,
                            bar_search_region, scales=_scales,
                            pre_gray=_bg, pre_offset=(_box, _boy),
                        )
                        return r, self.detector._last_scale

                    fut_fish = self._pool.submit(_detect_fish)
                    fut_bar = self._pool.submit(_detect_bar)
                    fish_result = fut_fish.result()
                    bar_result = fut_bar.result()

                    fish, _matched_key = fish_result
                    bar, _bar_scale = bar_result
                if not _use_yolo:
                    fish_detect_name = ""
                    if locked_fish_key:
                        if fish is not None:
                            fish_detect_name = locked_fish_key
                        if (fish is None and fish_lost > 20
                                and fish_lost % 20 == 0):
                            locked_fish_key = None
                            locked_fish_scales = None
                            log.info("  ★ Unlock fish template lock, search again")
                    else:
                        if fish is not None:
                            fish_detect_name = _matched_key or "?"
                            if (_matched_key
                                    and _matched_key != "fish_white"):
                                locked_fish_key = _matched_key
                                s = self.detector._last_best_scale
                                locked_fish_scales = [
                                    round(s * 0.85, 2), s,
                                    round(s * 1.15, 2)
                                ]
                                log.info(
                                    f"  ★ Lock fish template: "
                                    f"{locked_fish_key} @ scales="
                                    f"{[f'{x:.2f}' for x in locked_fish_scales]}"
                                )

                if fish is not None:
                    self._current_fish_name = fish_detect_name
                    if not _skip_fish and fish_detect_name:
                        wl_key = fish_detect_name
                        if not config.FISH_WHITELIST.get(wl_key, True):
                            fname_en = self.FISH_DISPLAY.get(
                                wl_key, (wl_key,))[0]
                            log.info(
                                f"[Whitelist] {fname_en} not in whitelist, give up this fish")
                            _skip_fish = True

                if not _use_yolo and bar is not None and not locked_bar_scales:
                    locked_bar_scales = [
                        round(max(0.2, _bar_scale * 0.85), 2),
                        _bar_scale,
                        round(_bar_scale * 1.15, 2),
                    ]
                    log.info(
                        f"  ★ Lock bar "
                        f"@ scales={[f'{x:.2f}' for x in locked_bar_scales]}"
                    )

                # ════════════ ★ X-axis verification (fish and bar share track X) ════════════
                if bar is not None:
                    raw_bcx = bar[0] + bar[2] // 2
                    if self._bar_locked_cx is None:
                        self._bar_locked_cx = raw_bcx
                        log.info(f"  ★ Track X-axis locked (bar): X={raw_bcx}")
                    elif abs(raw_bcx - self._bar_locked_cx) > _BAR_X_HALF:
                        bar = None
                    if bar is not None:
                        bar = (self._bar_locked_cx - bar[2] // 2,
                               bar[1], bar[2], bar[3], bar[4])

                # ════════════ ★ First bar detection → lock Y-axis search range ════════════
                if bar is not None and not _regions_locked:
                    bar_cy = bar[1] + bar[3] // 2
                    tcx = self._bar_locked_cx or (bar[0] + bar[2] // 2)
                    y_top = max(0, bar_cy - config.REGION_UP)
                    y_bot = min(h_scr, bar_cy + config.REGION_DOWN)
                    _roi = config.DETECT_ROI
                    if _roi:
                        y_top = max(y_top, _roi[1])
                        y_bot = min(y_bot, _roi[1] + _roi[3])
                    rh = y_bot - y_top
                    # Fish: slightly wider search region than bar
                    fish_half = max(config.REGION_X * 2, 80)
                    fsx = max(0, tcx - fish_half)
                    fsw = min(fish_half * 2, w_scr - fsx)
                    if _roi:
                        fsx = max(fsx, _roi[0])
                        fsw = min(fsw, _roi[0] + _roi[2] - fsx)
                    search_region = (fsx, y_top, fsw, rh)
                    # Bar: tight search region (user controlled)
                    bar_half = config.REGION_X
                    bsx = max(0, tcx - bar_half)
                    bsw = min(bar_half * 2, w_scr - bsx)
                    if _roi:
                        bsx = max(bsx, _roi[0])
                        bsw = min(bsw, _roi[0] + _roi[2] - bsx)
                    bar_search_region = (bsx, y_top, bsw, rh)
                    _regions_locked = True
                    log.info(
                        f"  ★ Search region locked (bar Y={bar_cy}): "
                        f"Y={y_top}~{y_bot} "
                        f"fish X=±{fish_half} bar X=±{bar_half}"
                        f"{' (ROI crop)' if _roi else ''}"
                    )

                # Fish: verify with same track X, discard if deviation too large
                if fish is not None:
                    raw_fcx = fish[0] + fish[2] // 2
                    if self._bar_locked_cx is not None:
                        if abs(raw_fcx - self._bar_locked_cx) > _FISH_X_HALF:
                            fish = None
                            self._current_fish_name = ""
                    if fish is not None and self._bar_locked_cx is not None:
                        fish = (self._bar_locked_cx - fish[2] // 2,
                                fish[1], fish[2], fish[3], fish[4])

                # ════════════ ★ Spatial validity check (Y-axis only) ════════════
                if fish is not None and bar is not None:
                    fish_cy_check = fish[1] + fish[3] // 2
                    bar_cy_check  = bar[1]  + bar[3]  // 2
                    dist_y = abs(fish_cy_check - bar_cy_check)

                    if dist_y > config.MAX_FISH_BAR_DIST:
                        if frame % 30 == 1:
                            log.warning(
                                f"[⚠ Misdetect] fish Y={fish_cy_check} bar Y="
                                f"{bar_cy_check} distance={dist_y}px > "
                                f"{config.MAX_FISH_BAR_DIST}px"
                            )
                        fish = None
                        bar = None

                # ════════════ ★ Visual debug (draw each frame, built-in throttle) ════════════
                # ★ Show original screen (no rotation), more intuitive
                # (rotation coords slightly off, but better than rotated view)
                if not self._need_rotation:
                    self._show_debug_overlay(
                        screen_raw, fish, bar, search_region,
                        bar_search_region=bar_search_region,
                        progress=_yolo_progress,
                        status_text=f"🐟 Minigame F{frame:04d}"
                    )
                else:
                    self._show_debug_overlay(
                        screen_raw,
                        bar_search_region=bar_search_region,
                        progress=_yolo_progress,
                        status_text=f"🐟 Minigame F{frame:04d} (rotation {self._track_angle:.0f}° compensating)"
                    )

                # ════════════ Progress bar (track progress, don't directly end) ════════════
                green = 0.0
                if frame <= _PROGRESS_SKIP_FRAMES:
                    pass
                elif _use_yolo and _yolo_progress is not None:
                    px, py, pw, ph = _yolo_progress[:4]
                    pcx = px + pw // 2
                    strip_w = 5
                    sx = max(0, pcx - strip_w // 2)
                    green = self.detector.detect_green_ratio(
                        screen, (sx, py, strip_w, ph))
                    if not self._progress_debug_saved and green > 0:
                        self._progress_debug_saved = True
                        _pad = 20
                        _dx = max(0, px - _pad)
                        _dw = min(pw + _pad * 2, w_scr - _dx)
                        _dbg = screen[py:py + ph, _dx:_dx + _dw].copy()
                        cv2.rectangle(_dbg, (sx - _dx, 0),
                                      (sx - _dx + strip_w, ph),
                                      (0, 255, 0), 1)
                        _info = f"green={green:.0%} w={strip_w}"
                        cv2.putText(_dbg, _info, (2, 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 255), 1)
                        _ddir = os.path.join(config.BASE_DIR, "debug")
                        os.makedirs(_ddir, exist_ok=True)
                        cv2.imwrite(
                            os.path.join(_ddir, "progress_strip.png"), _dbg)
                else:
                    _sr_for_progress = search_region
                    if bar is not None:
                        bcx = bar[0] + bar[2] // 2
                        bcy = bar[1] + bar[3] // 2
                        _pr_half_x = max(config.REGION_X * 2, 80)
                        _pr_x = max(0, bcx - _pr_half_x)
                        _pr_y = max(0, bcy - config.REGION_UP)
                        _pr_w = min(_pr_half_x * 2, w_scr - _pr_x)
                        _pr_h = min(config.REGION_UP + config.REGION_DOWN,
                                    h_scr - _pr_y)
                        _sr_for_progress = (_pr_x, _pr_y, _pr_w, _pr_h)
                        _last_progress_sr = _sr_for_progress
                    elif _last_progress_sr is not None:
                        _sr_for_progress = _last_progress_sr
                    green = self._check_progress(
                        screen, fish, _sr_for_progress)

                if green > 0 and _prev_green > 0.01 and (green - _prev_green) > 0.30:
                    log.debug(f"  Progress jump too large {_prev_green:.0%}→{green:.0%}, ignore")
                    green = _prev_green

                if green > 0:
                    _prev_green = green
                if green > _last_green:
                    _last_green = green

                # ════════════ Game end detection ════════════
                # ★ Count objects detected this frame (fish/bar/track)
                obj_count = ((fish is not None) + (bar is not None)
                             + (1 if track_alive else 0))

                # 1) Neither fish nor bar detected → count
                if fish is None and bar is None:
                    no_detect += 1
                    if no_detect > 5 and not config.IL_RECORD:
                        self.input.mouse_up()

                    if no_detect == 10:
                        log.warning(
                            f"[⚠ Missing] {no_detect} consecutive frames fish+bar undetected"
                        )
                        self.screen.save_debug(screen, "minigame_lost")

                    if no_detect > config.TRACK_LOST_LIMIT:
                        log.info(f"[📋 End] {no_detect} frames no valid UI detected, game ended")
                        break

                    time.sleep(config.GAME_LOOP_INTERVAL)
                    continue
                else:
                    if no_detect > 5:
                        log.info(f"[✓ Recovered] Valid UI re-detected (missed {no_detect} frames)")
                    no_detect = 0

                # 2) Separately track fish disappearance (bar may misdetect)
                if fish is None:
                    fish_lost += 1
                    if fish_gone_since is None:
                        fish_gone_since = time.time()
                    if fish_lost == 30:
                        log.warning(f"[⚠ Fish lost] {fish_lost} consecutive frames fish undetected")
                    if had_good_detection and fish_lost > config.FISH_LOST_LIMIT:
                        log.info(f"[📋 End] Fish missing {fish_lost} frames, game may have ended")
                        break
                else:
                    fish_lost = 0
                    fish_gone_since = None
                    had_good_detection = True

                if bar is None:
                    if bar_gone_since is None:
                        bar_gone_since = time.time()
                else:
                    bar_gone_since = None

                # ★ Single object timeout: either fish or bar missing > N seconds → fail
                _timeout = config.SINGLE_OBJ_TIMEOUT
                now_t = time.time()
                if (had_good_detection and fish_gone_since is not None
                        and now_t - fish_gone_since > _timeout):
                    elapsed = now_t - fish_gone_since
                    log.info(
                        f"[📋 Failed] Fish missing {elapsed:.1f}s "
                        f"(>{_timeout}s), game ended"
                    )
                    break
                if (had_good_detection and bar_gone_since is not None
                        and now_t - bar_gone_since > _timeout):
                    elapsed = now_t - bar_gone_since
                    log.info(
                        f"[📋 Failed] Bar missing {elapsed:.1f}s "
                        f"(>{_timeout}s), game ended"
                    )
                    break

                # 3) ★ Insufficient object detection: need min 2 of fish/bar/track
                if obj_count < config.OBJ_MIN_COUNT:
                    obj_gone_count += 1
                    if obj_gone_count == 1 or obj_gone_count % 10 == 0:
                        has_f = "fish✓" if fish is not None else "fish✗"
                        has_b = "bar✓" if bar is not None else "bar✗"
                        has_t = "track✓" if track_alive else "track✗"
                        log.warning(
                            f"[⚠ Insufficient objects] {has_f} {has_b} {has_t} "
                            f"= {obj_count} objects "
                            f"({obj_gone_count}/{config.OBJ_GONE_LIMIT})"
                        )
                    if obj_gone_count >= config.OBJ_GONE_LIMIT:
                        log.info(
                            f"[📋 End] {obj_gone_count} frames detected only "
                            f"{obj_count} objects, game ended!"
                        )
                        break
                else:
                    if obj_gone_count > 3:
                        log.info(
                            f"[✓ Recovered] Object count back to {obj_count} "
                            f"(insufficient for {obj_gone_count} frames)"
                        )
                    obj_gone_count = 0

                # ════════════ ★ Control (record / model / PD) ════════════
                if _skip_fish:
                    self.input.mouse_up()
                    held = False
                elif config.IL_RECORD:
                    self._il_record_frame(frame, fish, bar)
                    held = False
                elif config.IL_USE_MODEL and self._il_policy is not None:
                    held = self._il_model_control(fish, bar)
                else:
                    held = self._control_mouse(fish, bar, search_region)
                if held:
                    hold_count += 1

                # After 5s switch to user debug mode
                if frame == 50:
                    self.detector.debug_report = self.debug_mode

                # ── Logging (output every 30 frames) ──
                if frame % 30 == 0:
                    fname = self._current_fish_name.replace(
                        "fish_", ""
                    ) if self._current_fish_name else ""
                    fi = (f"fish[{fname}]Y={fish[1]+fish[3]//2}"
                          if fish else "fish=none")
                    bi = f"bar Y={bar[1]+bar[3]//2}" if bar else "bar=none"
                    vel = f"v={self._bar_velocity:+.0f}"
                    log.info(
                        f"[F{frame:04d}] {fi} | {bi} | {vel} | "
                        f"hold:{hold_count} | progress:{green:.0%}"
                    )

                time.sleep(config.GAME_LOOP_INTERVAL)

        finally:
            if _skip_fish:
                success = False
                log.info(
                    f"[⏭ Skipped] Non-target fish, given up (progress {_last_green:.0%} not counted)"
                )
            elif _last_green > config.SUCCESS_PROGRESS:
                success = True
                log.info(
                    f"[✅ Success] Final progress {_last_green:.0%} > "
                    f"{config.SUCCESS_PROGRESS:.0%}, judged success"
                )
            else:
                log.info(
                    f"[❌ Failed] Final progress {_last_green:.0%} <= "
                    f"{config.SUCCESS_PROGRESS:.0%}, judged failed"
                )

            if config.IL_RECORD:
                self._il_stop_recording()
                log.info("[🎣 Reel in] Record mode - reel in manually")
            else:
                self.input.safe_release()
                # Safety interval: prevent last mouse_down from becoming accidental click
                # when game ends (causes unintended cast)
                time.sleep(0.5)
                if success:
                    time.sleep(0.2)
                    self.input.click()
                    log.info("[🎣 Reel in] Fishing success, click to reel in")
                else:
                    log.info("[🎣 Failed] Rod already auto-retracted, skip reel in")

        return success

    # ══════════════════════════════════════════════════════
    #  Visual debug
    # ══════════════════════════════════════════════════════

    def _show_debug_overlay(self, screen, fish=None, bar=None,
                            search_region=None, bar_search_region=None,
                            progress=None, status_text=""):
        """
        Unified debug window - available for all stages.
        ★ Resize to small image first, then draw overlay, significantly reduces CPU/memory.
        """
        if not config.SHOW_DEBUG:
            return
        now = time.time()
        if now - self._last_overlay_time < config.DEBUG_OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now

        # ── ROI crop: only show selected region ──
        _roi = config.DETECT_ROI
        ox, oy = 0, 0
        if _roi:
            rx, ry, rw, rh = _roi
            sh, sw = screen.shape[:2]
            rx = max(0, min(rx, sw - 1))
            ry = max(0, min(ry, sh - 1))
            rw = min(rw, sw - rx)
            rh = min(rh, sh - ry)
            if rw > 20 and rh > 20:
                screen = screen[ry:ry + rh, rx:rx + rw].copy()
                ox, oy = rx, ry

        h, w = screen.shape[:2]
        max_w = config.DEBUG_OVERLAY_MAX_W
        max_h = config.DEBUG_OVERLAY_MAX_H
        s = min(max_w / w, max_h / h, 1.0)

        if s < 1.0:
            debug = cv2.resize(screen, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_NEAREST)
        else:
            debug = screen.copy()
            s = 1.0

        # ── Coordinate scaling helper (subtract ROI offset, then scale) ──
        def sx(v):
            return int((v - ox) * s)

        def sy(v):
            return int((v - oy) * s)

        # ── Top status text ──
        y_txt = 22
        fs = 0.55
        dw = debug.shape[1]
        # ★ FPS display (top right)
        fps_text = f"{self._fps:.1f} FPS"
        fps_color = (0, 255, 0) if self._fps >= 10 else (0, 255, 255) if self._fps >= 5 else (0, 0, 255)
        cv2.putText(debug, fps_text, (dw - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        if status_text:
            cv2.putText(debug, status_text, (8, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1)
            y_txt += 22

        if self._need_rotation:
            cv2.putText(debug, f"Rotation: {-self._track_angle:.1f} deg",
                        (8, y_txt), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 200, 255), 1)
            y_txt += 20

        # ★ Control state + velocity annotation
        if fish is not None and bar is not None:
            fish_cy = fish[1] + fish[3] // 2
            bar_cy  = bar[1]  + bar[3]  // 2
            diff = bar_cy - fish_cy
            if diff > config.DEAD_ZONE:
                label = f"v BAR below (d={diff}px)"
                lcolor = (0, 100, 255)
            elif diff < -config.DEAD_ZONE:
                label = f"^ BAR above (d={diff}px)"
                lcolor = (255, 200, 0)
            else:
                label = f"= dead zone (d={diff}px)"
                lcolor = (0, 255, 0)
            cv2.putText(debug, label, (8, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, lcolor, 1)
            y_txt += 20
        elif fish is None and bar is None and self.state == "Fishing minigame":
            cv2.putText(debug, "X no fish+bar", (8, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 1)
            y_txt += 20

        if abs(self._bar_velocity) > 0.5:
            cv2.putText(debug, f"v={self._bar_velocity:+.0f} px/s",
                        (8, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1)
            y_txt += 18

        # ── Draw search regions (gray=fish, light cyan=bar) ──
        if search_region:
            rx, ry, rw, rh = [int(v) for v in search_region]
            cv2.rectangle(debug, (sx(rx), sy(ry)),
                          (sx(rx + rw), sy(ry + rh)), (128, 128, 128), 1)
        if bar_search_region:
            bx, by, bw, bh = [int(v) for v in bar_search_region]
            cv2.rectangle(debug, (sx(bx), sy(by)),
                          (sx(bx + bw), sy(by + bh)), (128, 200, 200), 1)

        # ── Draw fish + show fish color name ──
        if fish is not None:
            fx, fy, fw, fh = fish[:4]
            fish_cy = fy + fh // 2
            fname, fcolor = self.FISH_DISPLAY.get(
                self._current_fish_name, ("?", (0, 255, 0))
            )
            cv2.rectangle(debug, (sx(fx), sy(fy)),
                          (sx(fx + fw), sy(fy + fh)), fcolor, 2)
            cv2.putText(debug, f"{fname} Y={fish_cy}",
                        (sx(fx + fw) + 4, sy(fish_cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, fcolor, 1)
            cv2.line(debug, (sx(fx), sy(fish_cy)),
                     (sx(fx + fw), sy(fish_cy)), fcolor, 1)

        # ── Draw bar (blue) ──
        if bar is not None:
            bx, by, bw, bh = bar[:4]
            bar_cy = by + bh // 2
            cv2.rectangle(debug, (sx(bx), sy(by)),
                          (sx(bx + bw), sy(by + bh)), (255, 100, 0), 2)
            cv2.putText(debug, f"Bar Y={bar_cy}",
                        (max(0, sx(bx) - 90), sy(bar_cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 100, 0), 1)
            cv2.line(debug, (sx(bx), sy(bar_cy)),
                     (sx(bx + bw), sy(bar_cy)), (255, 100, 0), 1)

        # ── Draw progress bar (yellow-green) ──
        if progress is not None:
            px, py, pw, ph = progress[:4]
            cv2.rectangle(debug, (sx(px), sy(py)),
                          (sx(px + pw), sy(py + ph)), (0, 220, 180), 2)
            cv2.putText(debug, "Progress",
                        (sx(px), sy(py) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 180), 1)

        # ── Connection line between fish and bar ──
        if fish is not None and bar is not None:
            fish_cy = fish[1] + fish[3] // 2
            bar_cy  = bar[1]  + bar[3]  // 2
            cx = (fish[0] + bar[0]) // 2
            diff = bar_cy - fish_cy
            color = (0, 0, 255) if abs(diff) > 50 else (0, 255, 255)
            cv2.arrowedLine(debug, (sx(cx), sy(bar_cy)),
                            (sx(cx), sy(fish_cy)), color, 1, tipLength=0.15)
            cv2.putText(debug, f"d={diff:+d}",
                        (sx(cx) + 6, sy((fish_cy + bar_cy) // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        with self._debug_lock:
            self._debug_frame = debug

        if self._debug_thread is None or not self._debug_thread.is_alive():
            self._debug_stop_event.clear()
            self._debug_thread = threading.Thread(
                target=self._debug_display_loop, daemon=True
            )
            self._debug_thread.start()

    def stop_debug_window(self, timeout=2.0):
        """
        ★ Safely close debug window (for GUI calls).
        Notify debug thread to exit via event, not directly call cv2.destroyWindow from GUI thread.
        This avoids OpenCV GUI cross-thread deadlock.
        """
        self._debug_stop_event.set()
        with self._debug_lock:
            self._debug_frame = None
        if self._debug_thread is not None and self._debug_thread.is_alive():
            self._debug_thread.join(timeout=timeout)
        self._debug_thread = None

    def _debug_display_loop(self):
        """
        Independent thread: loop display debug frames, cv2.waitKey blocking doesn't affect fishing thread.
        ★ Fix: check config.SHOW_DEBUG and _debug_stop_event, ensure safe exit.
        OpenCV GUI functions (imshow/destroyWindow/waitKey) only called in this thread.
        """
        while (self.running or self._debug_frame is not None) and config.SHOW_DEBUG:
            # Check stop signal (set on ROI selection or debug close)
            if self._debug_stop_event.is_set():
                break
            frame = None
            with self._debug_lock:
                if self._debug_frame is not None:
                    frame = self._debug_frame
                    self._debug_frame = None
            if frame is not None:
                try:
                    cv2.imshow("Debug Overlay", frame)
                except Exception:
                    break
            key = cv2.waitKey(30)
            if key == 27:  # ESC
                break
        try:
            cv2.destroyWindow("Debug Overlay")
        except Exception:
            pass
        # ★ Wait for OpenCV window to actually close
        cv2.waitKey(1)

    # ══════════════════════════════════════════════════════
    #  Minigame helper
    # ══════════════════════════════════════════════════════

    def _init_search_region(self, screen):
        """
        Initialize search region, return (region, track_center_x, bar_region).

        ★ If player set DETECT_ROI (selected region):
          - Only search track/bar inside ROI
          - ROI itself becomes initial search region
        ★ Without ROI: cross-validation (bar+track) positioning
        """
        h, w = screen.shape[:2]
        roi = config.DETECT_ROI

        # Validate ROI validity
        if roi:
            rx, ry, rw, rh = roi
            if rx + rw > w or ry + rh > h or rw < 20 or rh < 20:
                log.warning(
                    f"  ► ROI ({rx},{ry},{rw},{rh}) exceeds screen "
                    f"({w}x{h}) or too small, ignored"
                )
                roi = None

        # Search bar and track inside ROI (or fullscreen)
        bar = self.detector.find_multiscale(
            screen, "bar", config.THRESH_BAR,
            scales=config.BAR_SCALES,
            search_region=roi,
        )
        track = self.detector.find_multiscale(
            screen, "track", config.THRESH_TRACK,
            search_region=roi,
        )

        bar_cx = (bar[0] + bar[2] // 2) if bar else None
        track_cx = (track[0] + track[2] // 2) if track else None

        chosen_cx = None

        if bar_cx is not None and track_cx is not None:
            if abs(bar_cx - track_cx) < 150:
                chosen_cx = bar_cx
                log.info(
                    f"  ► Track+bar consistent: track X={track_cx}(conf={track[4]:.2f}) "
                    f"bar X={bar_cx}(conf={bar[4]:.2f}) → use bar X"
                )
            else:
                chosen_cx = bar_cx
                log.warning(
                    f"  ► Track X={track_cx}(conf={track[4]:.2f}) "
                    f"bar X={bar_cx}(conf={bar[4]:.2f}) inconsistent, "
                    f"use bar"
                )
        elif bar_cx is not None:
            chosen_cx = bar_cx
            log.info(f"  ► Only bar detected @ X={bar_cx} conf={bar[4]:.2f}")
        elif track_cx is not None:
            chosen_cx = track_cx
            log.info(f"  ► Only track detected @ X={track_cx} conf={track[4]:.2f}")

        # ── Has ROI → directly use ROI as search region ──
        if roi:
            roi_t = tuple(roi)
            if chosen_cx is None:
                chosen_cx = roi[0] + roi[2] // 2
                log.info(f"  ► No track/bar in ROI, use ROI center X={chosen_cx}")
            log.info(
                f"  ★ Use selected region: X={roi[0]} Y={roi[1]} "
                f"{roi[2]}x{roi[3]}"
            )
            return roi_t, chosen_cx, roi_t

        # ── No ROI → build region from detection results ──
        if chosen_cx is not None:
            y_start = h // 3
            bar_half = max(config.REGION_X, 60)
            bsx = max(0, chosen_cx - bar_half)
            bsw = min(bar_half * 2, w - bsx)
            bar_region = (bsx, y_start, bsw, h - y_start)
            fish_half = max(config.REGION_X * 2, 120)
            fsx = max(0, chosen_cx - fish_half)
            fsw = min(fish_half * 2, w - fsx)
            fish_region = (fsx, y_start, fsw, h - y_start)
            return fish_region, chosen_cx, bar_region

        sw = int(w * 0.6)
        y_start = h // 2
        log.info("  ► No track/bar found, use left lower region")
        fallback = (0, y_start, sw, h - y_start)
        return fallback, None, fallback

    _progress_debug_saved = False

    def _check_progress(self, screen, fish, sr):
        """
        Detect progress bar (green part).
        Detect green ratio in narrow strip 5px wide left of bar center X.
        """
        if sr is None:
            return 0.0

        bar_cx = self._bar_locked_cx
        if bar_cx is None:
            if fish is not None:
                bar_cx = fish[0]
            else:
                bar_cx = sr[0] + sr[2] // 3

        strip_w = 5
        sx = max(0, bar_cx - strip_w - 8)
        sy = sr[1]
        sw = strip_w
        sh = sr[3]
        if sx + sw > screen.shape[1]:
            sw = screen.shape[1] - sx
        if sy + sh > screen.shape[0]:
            sh = screen.shape[0] - sy
        if sw <= 0 or sh <= 0:
            return 0.0

        ratio = self.detector.detect_green_ratio(
            screen, (sx, sy, sw, sh))

        if not self._progress_debug_saved and ratio > 0:
            self._progress_debug_saved = True
            import os
            pad = 30
            dx = max(0, sx - pad)
            dw = min(sw + pad * 2, screen.shape[1] - dx)
            dbg = screen[sy:sy + sh, dx:dx + dw].copy()
            cv2.rectangle(dbg, (sx - dx, 0), (sx - dx + sw, sh),
                          (0, 255, 0), 1)
            info = f"green={ratio:.0%} w={strip_w}"
            cv2.putText(dbg, info, (2, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            debug_dir = os.path.join(config.BASE_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, "progress_strip.png"), dbg)

        return ratio

    # ══════════════════════════════════════════════════════
    #  Imitation learning: record / inference
    # ══════════════════════════════════════════════════════

    def _load_il_policy(self):
        """Load trained imitation learning model (with normalization params)"""
        try:
            import torch
            from imitation.model import FishPolicy
            checkpoint = torch.load(config.IL_MODEL_PATH, map_location="cpu",
                                    weights_only=True)

            # Compatible with old format (pure state_dict) and new format (with normalization)
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
                self._il_norm_mean = checkpoint["norm_mean"].numpy()
                self._il_norm_std = checkpoint["norm_std"].numpy()
                hist_len = checkpoint.get("history_len", config.IL_HISTORY_LEN)
            else:
                state = checkpoint
                self._il_norm_mean = None
                self._il_norm_std = None
                hist_len = config.IL_HISTORY_LEN

            model = FishPolicy(history_len=hist_len)
            model.load_state_dict(state)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                self._il_device = "cuda"
            self._il_policy = model
            norm_info = "with normalization" if self._il_norm_mean is not None else "no normalization"
            log.info(f"[IL] Model loaded ({self._il_device}, {norm_info})")
        except Exception as e:
            log.warning(f"[IL] Model load failed: {e}")
            self._il_policy = None

    def _il_start_recording(self):
        """Start recording one minigame session data"""
        os.makedirs(config.IL_DATA_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.IL_DATA_DIR, f"session_{ts}.csv")
        self._il_file = open(path, "w", newline="", encoding="utf-8")
        self._il_writer = csv.writer(self._il_file)
        self._il_writer.writerow([
            "frame", "timestamp",
            "fish_cy", "bar_cy", "bar_h",
            "error", "velocity", "fish_delta", "dist_ratio",
            "mouse_pressed",
            "fish_in_bar", "press_streak",
            "predicted", "bar_accel",
        ])
        self._il_prev_fish_cy = None
        self._il_mouse_prev = 0
        self._il_history.clear()
        log.info(f"[IL] Recording started → {path}")

    def _il_stop_recording(self):
        """Stop recording"""
        if self._il_file:
            self._il_file.close()
            self._il_file = None
            self._il_writer = None
            log.info("[IL] Recording stopped")

    @staticmethod
    def _is_mouse_pressed() -> bool:
        """Check if user is pressing left mouse button"""
        return ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000 != 0

    def _il_build_features(self, fish, bar):
        """Build one frame features from detection results [10D]"""
        fish_cy = fish[1] + fish[3] // 2
        bar_cy = bar[1] + bar[3] // 2
        bar_h = bar[3]
        bar_top = bar[1]
        error = bar_cy - fish_cy
        velocity = self._bar_velocity
        fish_delta = 0.0
        if self._il_prev_fish_cy is not None:
            fish_delta = fish_cy - self._il_prev_fish_cy
        self._il_prev_fish_cy = fish_cy
        dist_ratio = error / max(bar_h, 1)

        fish_in_bar = (fish_cy - bar_top) / max(bar_h, 1)

        if self._il_mouse_prev == 1:
            self._il_press_streak = max(1, getattr(self, '_il_press_streak', 0) + 1)
        else:
            self._il_press_streak = min(-1, getattr(self, '_il_press_streak', 0) - 1)
        press_streak = self._il_press_streak / 10.0

        # Inertia prediction: bar relative to fish after 150ms
        predicted = error + velocity * 0.15

        # Acceleration: velocity change
        bar_accel = 0.0
        if hasattr(self, '_il_prev_velocity'):
            bar_accel = velocity - self._il_prev_velocity
        self._il_prev_velocity = velocity

        return [error, velocity, bar_h, fish_delta, dist_ratio,
                self._il_mouse_prev, fish_in_bar, press_streak,
                predicted, bar_accel]

    def _il_record_frame(self, frame_idx, fish, bar):
        """Record one frame: read user mouse state and write to CSV"""
        if fish is None or bar is None or self._il_writer is None:
            return

        mouse = 1 if self._is_mouse_pressed() else 0
        feats = self._il_build_features(fish, bar)
        fish_cy = fish[1] + fish[3] // 2
        bar_cy = bar[1] + bar[3] // 2
        bar_h = bar[3]
        error = feats[0]
        velocity = feats[1]
        fish_delta = feats[3]
        dist_ratio = feats[4]

        fish_in_bar = feats[6]
        press_streak = feats[7]
        predicted = feats[8]
        bar_accel = feats[9]

        self._il_writer.writerow([
            frame_idx, f"{time.time():.4f}",
            fish_cy, bar_cy, bar_h,
            f"{error:.1f}", f"{velocity:.1f}", f"{fish_delta:.1f}",
            f"{dist_ratio:.3f}",
            mouse,
            f"{fish_in_bar:.3f}", f"{press_streak:.2f}",
            f"{predicted:.1f}", f"{bar_accel:.1f}",
        ])
        self._il_mouse_prev = mouse

    def _il_model_control(self, fish, bar) -> bool:
        """
        Use trained model to decide press/release - state-based control (not pulse).
        Model output = "mouse should be pressed or released now", same as during recording.
        """
        import torch

        if self._il_policy is None:
            return False

        if fish is not None and bar is not None:
            feats = self._il_build_features(fish, bar)
            self._il_history.append(feats)
        elif fish is None and bar is None:
            self.input.mouse_up()
            self._il_mouse_prev = 0
            return False

        if len(self._il_history) < config.IL_HISTORY_LEN:
            self.input.mouse_down()
            self._il_mouse_prev = 1
            return True

        import numpy as np
        flat = []
        for f in self._il_history:
            flat.extend(f)
        flat_np = np.array(flat, dtype=np.float32)
        if self._il_norm_mean is not None:
            flat_np = (flat_np - self._il_norm_mean) / self._il_norm_std
        x = torch.from_numpy(flat_np).unsqueeze(0).to(self._il_device)
        prob = self._il_policy.predict(x)

        fish_cy = fish[1] + fish[3] // 2 if fish else -1
        bar_cy = bar[1] + bar[3] // 2 if bar else -1

        thresh = config.IL_PRESS_THRESH
        if prob > thresh:
            self.input.mouse_down()
            self._il_mouse_prev = 1
            if fish is not None and bar is not None and self._il_log_counter % 10 == 0:
                log.info(
                    f"  [IL] fish Y={fish_cy} bar Y={bar_cy} "
                    f"p={prob:.2f}>{thresh:.2f} → press"
                )
            self._il_log_counter += 1
            return True
        else:
            self.input.mouse_up()
            self._il_mouse_prev = 0
            if fish is not None and bar is not None and self._il_log_counter % 10 == 0:
                log.info(
                    f"  [IL] fish Y={fish_cy} bar Y={bar_cy} "
                    f"p={prob:.2f}<={thresh:.2f} → release"
                )
            self._il_log_counter += 1
            return False

    def _control_mouse(self, fish, bar, sr) -> bool:
        """
        PD physical controller (Stardew Valley fishing):

        Physics model:
        - Press mouse → bar gets upward acceleration, longer press = faster speed
        - Release mouse → gravity decelerates bar → stops → accelerates down
        - Bar has inertia: continues moving same direction even after release

        Control strategy:
        - Calculate 'error' = bar center - fish center (positive = bar below)
        - Estimate 'velocity' = bar movement speed (positive = down, negative = up)
        - Use velocity to predict future position, release early to avoid inertia overshoot
        - Hold duration ∝ predicted error (far = long hold, near = short hold)

        Return: whether press operation was executed
        """
        now = time.time()

        # ═══════════ ★ Velocity estimation: update whenever bar detected ═══════════
        if bar is not None:
            bar_cy_raw = bar[1] + bar[3] // 2
            if (self._bar_prev_cy is not None
                    and self._bar_prev_time is not None):
                dt = now - self._bar_prev_time
                if dt > 0.003:
                    raw_vel = (bar_cy_raw - self._bar_prev_cy) / dt
                    α = min(config.VELOCITY_SMOOTH, 0.95)
                    self._bar_velocity = (
                        α * self._bar_velocity + (1 - α) * raw_vel
                    )
            self._bar_prev_cy = bar_cy_raw
            self._bar_prev_time = now

        vel = self._bar_velocity

        # ═══════════ ★ Continuous PD controller (read GUI params) ═══════════
        TARGET_FIB = 0.5
        KP         = getattr(config, 'HOLD_GAIN', 0.040)
        KD         = getattr(config, 'SPEED_DAMPING', 0.00025)
        BASE_HOLD  = getattr(config, 'HOLD_MIN_S', 0.025)
        MAX_HOLD   = getattr(config, 'HOLD_MAX_S', 0.100)
        MIN_HOLD   = 0.004

        fast_lock_enabled = bool(getattr(config, "FAST_LOCK_ENABLED", True))

        if fish is not None and bar is not None:
            raw_fish_cy = fish[1] + fish[3] // 2
            bar_cy      = bar[1]  + bar[3]  // 2

            fish_vel = 0.0
            if (self._fish_prev_cy is not None
                    and self._fish_prev_time is not None):
                fish_dt = now - self._fish_prev_time
                if fish_dt > 0.003:
                    fish_vel = (raw_fish_cy - self._fish_prev_cy) / fish_dt
            self._fish_prev_cy = raw_fish_cy
            self._fish_prev_time = now

            # ── Fish position smoothing (EMA) ──
            if self._fish_smooth_cy is None:
                self._fish_smooth_cy = float(raw_fish_cy)
            else:
                alpha = 0.4
                if fast_lock_enabled:
                    jump_px = abs(raw_fish_cy - self._fish_smooth_cy)
                    jump_thr = max(1, int(getattr(config, "FAST_LOCK_JUMP_PX", 16)))
                    speed_thr = float(getattr(config, "FAST_LOCK_SPEED_PX_S", 700.0))
                    if jump_px >= jump_thr:
                        alpha = 0.80
                    elif abs(fish_vel) >= speed_thr:
                        alpha = 0.65
                self._fish_smooth_cy = (
                    alpha * raw_fish_cy + (1 - alpha) * self._fish_smooth_cy
                )
            fish_cy = int(self._fish_smooth_cy)

            bar_h   = max(bar[3], 1)
            bar_top = bar[1]
            fish_in_bar = (fish_cy - bar_top) / bar_h

            # Predict relative displacement: fish speed - bar speed
            predict_s = 0.0
            if fast_lock_enabled:
                predict_s = float(getattr(config, "FAST_LOCK_LOOKAHEAD_S", 0.040))
                predict_s = max(0.0, min(predict_s, 0.20))
            rel_vel = fish_vel - vel
            fish_in_bar_pred = fish_in_bar + rel_vel * predict_s / bar_h

            error_now = TARGET_FIB - fish_in_bar
            error_pred = TARGET_FIB - fish_in_bar_pred
            if fast_lock_enabled:
                # In fast lock, prioritize predicted error, reduce "chasing tail"
                error = 0.35 * error_now + 0.65 * error_pred
            else:
                error = error_now
            error_clamp = max(-2.0, min(2.0, error))

            fast_trigger = False
            if fast_lock_enabled:
                trigger_err = float(getattr(config, "FAST_LOCK_TRIGGER_ERR", 0.15))
                speed_thr = float(getattr(config, "FAST_LOCK_SPEED_PX_S", 700.0))
                fast_trigger = (
                    abs(error_now) >= trigger_err
                    or abs(error_pred) >= trigger_err
                    or abs(fish_vel) >= speed_thr
                )

            # hold = baseline + position correction + speed damping
            # vel>0(falling)→add hold to slow down; vel<0(rising)→reduce hold to prevent overshoot
            hold = BASE_HOLD + error_clamp * KP + vel * KD

            if fast_trigger:
                boost_gain = float(getattr(config, "FAST_LOCK_BOOST_GAIN", 0.050))
                boost_max = float(getattr(config, "FAST_LOCK_BOOST_MAX_S", 0.060))
                drop_err = float(getattr(config, "FAST_LOCK_DROP_ERR", 0.28))
                extra = min(
                    boost_max,
                    max(0.0, abs(error_clamp) - 0.08) * boost_gain
                )
                if error_clamp > 0:
                    hold += extra
                elif error_clamp < -drop_err:
                    # When fish rapidly drops below bar, actively release to let bar drop and chase
                    hold = 0.0
                else:
                    hold -= extra * 0.5

            if fast_lock_enabled:
                hold = max(0.0, min(hold, MAX_HOLD))
            else:
                hold = max(MIN_HOLD, min(hold, MAX_HOLD))

            # Record last state for fallback use
            self._last_hold = hold
            self._last_fish_cy = fish_cy

            fname = (self._current_fish_name.replace("fish_", "")
                     if self._current_fish_name else "?")

            if hold >= MIN_HOLD + 0.001:
                self.input.mouse_down()
                time.sleep(hold)
                self.input.mouse_up()
                log.info(
                    f"  ● [{fname}] fib={fish_in_bar:.2f} "
                    f"fv={fish_vel:+.0f} v={vel:+.0f}"
                    f"{' ⚡' if fast_trigger else ''} → press {hold*1000:.0f}ms"
                )
                return True
            else:
                self.input.mouse_up()
                log.info(
                    f"  ○ [{fname}] fib={fish_in_bar:.2f} "
                    f"fv={fish_vel:+.0f} v={vel:+.0f}"
                    f"{' ⚡' if fast_trigger else ''} → release"
                )
                return False

        # ── Fallback: only fish or only bar → use last hold decaying to baseline ──
        fallback = self._last_hold
        if fallback is None:
            fallback = BASE_HOLD

        # Decay: when no complete detection, trend toward baseline hover each frame
        fallback = 0.6 * fallback + 0.4 * BASE_HOLD
        self._last_hold = fallback

        if fish is not None:
            fish_cy = fish[1] + fish[3] // 2
            self._last_fish_cy = fish_cy
            # Fish above (need press) or below (need release)
            if sr is not None:
                mid_y = sr[1] + sr[3] // 2
            elif config.DETECT_ROI:
                mid_y = config.DETECT_ROI[1] + config.DETECT_ROI[3] // 2
            else:
                mid_y = fish_cy
            if fish_cy < mid_y:
                h = min(fallback * 1.5, MAX_HOLD)
                self.input.mouse_down()
                time.sleep(h)
                self.input.mouse_up()
                log.info(
                    f"  (only fish) Y={fish_cy} v={vel:+.0f}"
                    f" → press {h*1000:.0f}ms"
                )
                return True
            else:
                self.input.mouse_up()
                return False

        elif bar is not None:
            bar_cy = bar[1] + bar[3] // 2
            # Use last fish position to estimate fish_in_bar
            if self._last_fish_cy is not None:
                est_fib = (self._last_fish_cy - bar[1]) / max(bar[3], 1)
                error = TARGET_FIB - est_fib
                error_clamp = max(-2.0, min(2.0, error))
                hold = BASE_HOLD + error_clamp * KP + vel * KD
                hold = max(MIN_HOLD, min(hold, MAX_HOLD))
            else:
                hold = fallback
            self.input.mouse_down()
            time.sleep(hold)
            self.input.mouse_up()
            log.info(
                f"  (only bar) Y={bar_cy} v={vel:+.0f}"
                f" → press {hold*1000:.0f}ms"
            )
            return True

        return False

    # ══════════════════════════════════════════════════════
    #  Main loop (runs in background thread)
    # ══════════════════════════════════════════════════════

    def run(self):
        """Main fishing loop - started by GUI in background thread"""
        log.info("Fishing thread started")

        while self.running:
            try:
                if config.IL_RECORD:
                    # ★ Record mode: user manual operation, program waits for minigame UI
                    self.state = "Recording: waiting for minigame"
                    log.info("[IL] Manually cast → wait → hook, program waiting for minigame...")
                    if not self._wait_for_minigame_ui():
                        break
                else:
                    self._cast_rod()
                    if not self.running:
                        break

                    if not self._wait_for_bite():
                        if self.running:
                            time.sleep(1.0)
                        continue
                    if not self.running:
                        break

                    self._hook_fish()
                    if not self.running:
                        break

                    # ★ Verify minigame actually appeared
                    if not self._verify_minigame():
                        wait = config.POST_CATCH_DELAY
                        log.info(f"[🔄 Retry] Minigame not detected, wait {wait:.1f}s after reeling and recast")
                        self.input.click()
                        time.sleep(wait)
                        log.info("─" * 40)
                        continue

                if not self.running:
                    break

                result = self._fishing_minigame()

                self.fish_count += 1
                tag = "Success ✅" if result else "Complete"
                log.info(f"[🎣 Result] Fishing #{self.fish_count} — {tag}")
                log.info("─" * 40)

                self.state = "Waiting for next round"
                time.sleep(config.POST_CATCH_DELAY)

            except Exception as e:
                log.error(f"Runtime error: {e}")
                if not config.IL_RECORD:
                    self.input.safe_release()
                time.sleep(2)

        if not config.IL_RECORD:
            self.input.safe_release()
        self.state = "Stopped"
        log.info("Fishing thread stopped")
        try:
            cv2.destroyWindow("Debug Overlay")
        except Exception:
            pass
