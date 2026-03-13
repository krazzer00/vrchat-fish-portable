"""
Logging Utility Module
======================
Supports console output + GUI queue push + file save.
"""

import os
import time
import queue


class Logger:
    """Logger - console print + queue push + memory cache (for save)"""

    def __init__(self):
        self.log_queue: queue.Queue = queue.Queue()
        self._lines: list[str] = []

    def info(self, msg: str):
        self._emit("INFO", msg)

    def warning(self, msg: str):
        self._emit("WARN", msg)

    def error(self, msg: str):
        self._emit("ERROR", msg)

    def debug(self, msg: str):
        self._emit("DEBUG", msg)

    def _emit(self, level: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}][{level:>5s}] {msg}"
        print(line)
        self._lines.append(line)
        try:
            self.log_queue.put_nowait(line)
        except queue.Full:
            pass

    def save(self, path: str):
        """Overwrite all current logs to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._lines))
                f.write("\n")
        except Exception as e:
            print(f"[Logger] Failed to save logs: {e}")

    def clear(self):
        """Clear in-memory log cache"""
        self._lines.clear()


# Global singleton
log = Logger()
