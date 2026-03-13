"""
Window Management Module
========================
- Find VRChat window (fuzzy title matching)
- DPI-aware — ensures screenshot coordinates match actual pixels
- Smart focus — switches to foreground only when necessary
"""

import time
import ctypes
import ctypes.wintypes

from utils.logger import log

# ── DPI-aware (must call before any window operations) ──
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)          # Per-Monitor V2
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()           # Fallback
    except Exception:
        pass

# ── Win32 constants ──
SW_RESTORE = 9
GWL_STYLE = -16
WS_MINIMIZE = 0x20000000

user32 = ctypes.windll.user32


def _is_window(hwnd) -> bool:
    return bool(user32.IsWindow(hwnd))


def _is_iconic(hwnd) -> bool:
    return bool(user32.IsIconic(hwnd))


def _get_foreground() -> int:
    return user32.GetForegroundWindow()


class WindowManager:
    """VRChat Window Manager"""

    def __init__(self, title_keyword: str = "VRChat"):
        self.title_keyword = title_keyword
        self.hwnd = None
        self._title = ""
        self._rect = None       # (left, top, right, bottom)

    # ────────────────── Finding ──────────────────

    # Keywords to exclude script's own GUI window
    EXCLUDE_KEYWORDS = ["auto-fishing", "auto-fish", "auto fish"]

    def find(self) -> bool:
        """
        Enumerate all visible windows, match window with keyword in title.
        Auto-exclude script's own GUI window.
        """
        results = []
        keyword_lower = self.title_keyword.lower()
        exclude = [kw.lower() for kw in self.EXCLUDE_KEYWORDS]

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value
                    title_lower = title.lower()
                    # Must contain keyword and not match script's own GUI
                    if keyword_lower in title_lower:
                        if not any(ex in title_lower for ex in exclude):
                            results.append((hwnd, title))
            return True

        user32.EnumWindows(enum_cb, 0)

        if results:
            self.hwnd, self._title = results[0]
            self._update_rect()
            log.info(f"Found window: \"{self._title}\" (HWND={self.hwnd})")
            return True

        log.warning(f"No window containing \"{self.title_keyword}\" found (script self excluded)")
        self.hwnd = None
        return False

    # ────────────────── Focus ──────────────────

    def focus(self) -> bool:
        """
        Ensure VRChat is foreground window.
        Returns True directly if already foreground, no extra switching.
        """
        if not self.is_valid():
            if not self.find():
                return False

        # Already foreground → no operation needed
        if _get_foreground() == self.hwnd:
            return True

        try:
            if _is_iconic(self.hwnd):
                user32.ShowWindow(self.hwnd, SW_RESTORE)
                time.sleep(0.15)

            # Method 1: SetForegroundWindow
            user32.SetForegroundWindow(self.hwnd)
            time.sleep(0.1)

            if _get_foreground() == self.hwnd:
                return True

            # Method 2: Attach thread then retry
            fg_hwnd = _get_foreground()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None)
            my_tid = ctypes.windll.kernel32.GetCurrentThreadId()
            if fg_tid != my_tid:
                user32.AttachThreadInput(my_tid, fg_tid, True)
                user32.SetForegroundWindow(self.hwnd)
                user32.AttachThreadInput(my_tid, fg_tid, False)
                time.sleep(0.1)

            return _get_foreground() == self.hwnd

        except Exception as e:
            log.warning(f"Window focus failed: {e}")
            return False

    # ────────────────── Region ──────────────────

    def get_region(self):
        """
        Get window region on screen (x, y, w, h).
        Uses GetClientRect + ClientToScreen to get pure client area (no title bar/border).
        """
        if not self.is_valid():
            if not self.find():
                return None

        try:
            # Client area rect (relative to window top-left)
            rect = ctypes.wintypes.RECT()
            user32.GetClientRect(self.hwnd, ctypes.byref(rect))

            # Screen coordinates of client area top-left
            pt = ctypes.wintypes.POINT(0, 0)
            user32.ClientToScreen(self.hwnd, ctypes.byref(pt))

            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 0 and h > 0:
                return (pt.x, pt.y, w, h)
        except Exception:
            pass

        # Fallback: use GetWindowRect
        self._update_rect()
        if self._rect:
            l, t, r, b = self._rect
            if r - l > 0 and b - t > 0:
                return (l, t, r - l, b - t)
        return None

    # ────────────────── Status ──────────────────

    def is_valid(self) -> bool:
        return self.hwnd is not None and _is_window(self.hwnd)

    def is_foreground(self) -> bool:
        return self.is_valid() and _get_foreground() == self.hwnd

    @property
    def title(self) -> str:
        return self._title

    # ────────────────── Internal ──────────────────

    def _update_rect(self):
        if self.hwnd and _is_window(self.hwnd):
            try:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
                self._rect = (rect.left, rect.top, rect.right, rect.bottom)
            except Exception:
                self._rect = None
