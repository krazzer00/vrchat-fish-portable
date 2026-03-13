"""
Input Control Module
====================
PostMessage — Win32 message dispatch, no cursor move, no focus stealing.
Head shake via VRChat OSC API sends LookLeft/LookRight.
"""

import ctypes
import ctypes.wintypes
import time

from utils.logger import log

user32 = ctypes.windll.user32

WM_LBUTTONDOWN  = 0x0201
WM_LBUTTONUP    = 0x0202
WM_ACTIVATE     = 0x0006
WA_ACTIVE       = 1
MK_LBUTTON      = 0x0001


def _MAKELPARAM(x: int, y: int) -> int:
    return ((y & 0xFFFF) << 16) | (x & 0xFFFF)


class InputController:
    """PostMessage mouse controller + OSC head shake"""

    def __init__(self, window_mgr):
        self.wm = window_mgr
        self.mouse_is_down = False
        self._click_x = 400
        self._click_y = 400
        self._osc = None
        self._shake_dir = 1

    # ────────────────── Utilities ──────────────────

    def _update_click_pos(self):
        region = self.wm.get_region()
        if region:
            self._click_x = region[2] // 2
            self._click_y = region[3] // 2

    def _post(self, msg: int, wparam: int):
        hwnd = self.wm.hwnd
        if not hwnd:
            return False
        lparam = _MAKELPARAM(self._click_x, self._click_y)
        return bool(user32.PostMessageW(hwnd, msg, wparam, lparam))

    # ────────────────── Focus ──────────────────

    def focus_game(self) -> bool:
        ok = self.wm.focus()
        if ok:
            self._update_click_pos()
        else:
            log.warning("Failed to focus VRChat window")
        return ok

    def move_to_game_center(self):
        self._update_click_pos()

    def ensure_cursor_in_game(self):
        pass

    # ────────────────── Mouse Operations ──────────────────

    def click(self, focus: bool = False):
        if focus:
            self.focus_game()
            time.sleep(0.1)
        self._update_click_pos()
        self._post(WM_LBUTTONDOWN, MK_LBUTTON)
        time.sleep(0.06)
        self._post(WM_LBUTTONUP, 0)

    def click_rapid(self):
        self._post(WM_LBUTTONDOWN, MK_LBUTTON)
        time.sleep(0.02)
        self._post(WM_LBUTTONUP, 0)

    def mouse_down(self):
        if not self.mouse_is_down:
            self._post(WM_LBUTTONDOWN, MK_LBUTTON)
            self.mouse_is_down = True

    def mouse_up(self):
        if self.mouse_is_down:
            self._post(WM_LBUTTONUP, 0)
            self.mouse_is_down = False

    # ────────────────── Head Shake (OSC) ──────────────────

    def _get_osc(self):
        if self._osc is not None:
            return self._osc
        try:
            from pythonosc import udp_client
            self._osc = udp_client.SimpleUDPClient("127.0.0.1", 9000)
        except Exception:
            self._osc = None
        return self._osc

    def _osc_send(self, addr: str, value: int) -> bool:
        osc = self._get_osc()
        if osc is None:
            return False
        try:
            osc.send_message(addr, value)
            return True
        except Exception:
            # Auto-rebuild client on next send
            self._osc = None
            return False

    def _osc_reset_look(self, repeat: int = 2, interval: float = 0.01):
        for _ in range(max(1, int(repeat))):
            self._osc_send("/input/LookRight", 0)
            self._osc_send("/input/LookLeft", 0)
            if interval > 0:
                time.sleep(interval)

    def _osc_hold(self, addr: str, duration: float) -> bool:
        if not self._osc_send(addr, 1):
            return False
        time.sleep(duration)
        self._osc_send(addr, 0)
        return True

    def shake_head(self):
        """Head shake before cast, with forced reset and direction alternation, reduce long-term drift."""
        import config as _cfg
        t = getattr(_cfg, "SHAKE_HEAD_TIME", 0.01)
        if t <= 0:
            return
        gap = max(0.0, float(getattr(_cfg, "SHAKE_HEAD_GAP", 0.05)))
        reset_repeat = max(1, int(getattr(_cfg, "SHAKE_HEAD_RESET_REPEAT", 2)))
        reset_interval = max(0.0, float(getattr(_cfg, "SHAKE_HEAD_RESET_INTERVAL", 0.01)))

        if self._get_osc() is None:
            return

        seq = ("/input/LookRight", "/input/LookLeft")
        if self._shake_dir < 0:
            seq = ("/input/LookLeft", "/input/LookRight")

        try:
            self._osc_reset_look(reset_repeat, reset_interval)
            for i, addr in enumerate(seq):
                if not self._osc_hold(addr, t):
                    break
                if i < len(seq) - 1 and gap > 0:
                    time.sleep(gap)
            if gap > 0:
                time.sleep(gap)
        finally:
            self._osc_reset_look(reset_repeat, reset_interval)
            self._shake_dir *= -1

    # ────────────────── Safety ──────────────────

    def safe_release(self):
        try:
            self._post(WM_LBUTTONUP, 0)
        except Exception:
            pass
        try:
            self._osc_reset_look(2, 0.01)
        except Exception:
            pass
        self.mouse_is_down = False
