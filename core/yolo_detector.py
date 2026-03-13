"""
YOLO Object Detector
====================
Wraps ultralytics YOLO inference, provides interface compatible with template Detector.

Detection classes:
  0 = fish     (fish icon)           → returns (x, y, w, h, conf)
  1 = bar      (white catch bar)     → returns (x, y, w, h, conf)
  2 = track    (fishing track)       → returns (x, y, w, h, conf)
  3 = progress (green progress bar)  → returns (x, y, w, h, conf)
"""

import os
import cv2
import numpy as np
from utils.logger import log

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass


class YoloDetector:
    """YOLO-based fishing game detector."""

    CLASS_FISH = 0
    CLASS_BAR = 1
    CLASS_TRACK = 2
    CLASS_PROGRESS = 3

    def __init__(self, model_path: str, conf: float = 0.5, device="auto"):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.conf = conf
        self.model = YOLO(model_path)
        self._use_half = False   # FP16 half-precision flag

        import config as _cfg
        dev_pref = getattr(_cfg, "YOLO_DEVICE", "auto")
        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            pass
        if dev_pref == "cpu" or not cuda_ok:
            target_dev = "cpu"
        elif dev_pref == "gpu":
            target_dev = 0
        else:
            target_dev = 0

        # Optimization: use smaller inference size (configurable)
        self._imgsz = getattr(_cfg, "YOLO_IMGSZ", 640)

        warmup_img = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)

        if target_dev != "cpu":
            try:
                # Try FP16 half-precision (GPU speedup ~40-60%)
                self.model.predict(
                    warmup_img, conf=0.5, device=target_dev,
                    verbose=False, imgsz=self._imgsz, half=True,
                )
                self._use_half = True
                self._device = target_dev
                for _ in range(2):
                    self.model.predict(
                        warmup_img, conf=0.5, device=target_dev,
                        verbose=False, imgsz=self._imgsz, half=True,
                    )
                log.info(f"[YOLO] ✓ GPU FP16 mode ready (imgsz={self._imgsz}): {self.model.names}")
                return
            except Exception:
                # FP16 failed, try FP32
                try:
                    self.model.predict(
                        warmup_img, conf=0.5, device=target_dev,
                        verbose=False, imgsz=self._imgsz,
                    )
                    self._device = target_dev
                    for _ in range(2):
                        self.model.predict(
                            warmup_img, conf=0.5, device=target_dev,
                            verbose=False, imgsz=self._imgsz,
                        )
                    log.info(f"[YOLO] ✓ GPU FP32 mode ready (imgsz={self._imgsz}): {self.model.names}")
                    return
                except Exception as e:
                    if dev_pref == "gpu":
                        raise RuntimeError(f"[YOLO] Forced GPU mode but init failed: {e}")
                    log.warning(f"[YOLO] GPU unavailable ({e}), fallback to CPU")

        self._device = "cpu"
        self.model.predict(
            warmup_img, conf=0.5, device="cpu",
            verbose=False, imgsz=self._imgsz,
        )
        log.info(f"[YOLO] ✓ CPU mode ready (imgsz={self._imgsz}): {self.model.names}")

    def detect(self, screen, roi=None):
        """
        Run YOLO inference on one frame.

        Parameters:
            screen: BGR image (numpy array)
            roi:    [x, y, w, h] detection region (optional)

        Returns:
            dict: {
                'fish':  (x, y, w, h, conf) or None,
                'bar':   (x, y, w, h, conf) or None,
                'track': (x, y, w, h, conf) or None,
                'fish_name': str,  # Fish class name
                'raw': list,       # All detection results
            }
        """
        ox, oy = 0, 0
        img = screen

        if roi:
            rx, ry, rw, rh = roi
            h_s, w_s = screen.shape[:2]
            rx = max(0, min(rx, w_s))
            ry = max(0, min(ry, h_s))
            rw = min(rw, w_s - rx)
            rh = min(rh, h_s - ry)
            if rw > 10 and rh > 10:
                # Optimization: np.ascontiguousarray faster than .copy() (only copy when needed)
                img = np.ascontiguousarray(screen[ry:ry+rh, rx:rx+rw])
                ox, oy = rx, ry

        results = self.model.predict(
            img, conf=self.conf, device=self._device,
            verbose=False, imgsz=self._imgsz,
            half=self._use_half,
        )

        detections = {
            "fish": None,
            "bar": None,
            "track": None,
            "progress": None,
            "fish_name": "",
            "raw": [],
        }

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Optimization: batch convert to numpy/list avoid per-element .tolist()
        cls_arr = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()
        names = self.model.names

        for i in range(len(boxes)):
            cls = int(cls_arr[i])
            conf = float(conf_arr[i])
            x1, y1, x2, y2 = xyxy_arr[i]

            bx = int(x1) + ox
            by = int(y1) + oy
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            det = (bx, by, bw, bh, conf)
            class_name = names.get(cls, f"cls{cls}")
            detections["raw"].append((class_name, det))

            if class_name == "fish":
                if detections["fish"] is None or conf > detections["fish"][4]:
                    detections["fish"] = det
                    detections["fish_name"] = "fish"
            elif class_name == "bar":
                if detections["bar"] is None or conf > detections["bar"][4]:
                    detections["bar"] = det
            elif class_name == "track":
                if detections["track"] is None or conf > detections["track"][4]:
                    detections["track"] = det
            elif class_name == "progress":
                if detections["progress"] is None or conf > detections["progress"][4]:
                    detections["progress"] = det

        return detections

    def detect_track(self, screen, roi=None):
        """Detect only if track exists"""
        result = self.detect(screen, roi)
        return result["track"]

    def detect_bar(self, screen, roi=None):
        """Detect only white bar"""
        result = self.detect(screen, roi)
        return result["bar"]

    def detect_fish(self, screen, roi=None):
        """Detect only fish"""
        result = self.detect(screen, roi)
        return result["fish"], result["fish_name"]
