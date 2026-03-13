"""
GUI module (tkinter)
====================
Main control panel: status, buttons, parameter tuning,
real-time log output.
Bot runs in a background thread, GUI communicates via shared properties + log queue.

Features:
- i18n (RU/EN) with language selector
- translation of displayed bot state (bot.state) without modifying core.bot
- UI text rebuild on language change
- UI_LANG persistence in settings.json
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import json
import keyboard
import cv2

import config
from core.bot import FishingBot
from utils.logger import log


# ═══════════════════════════════════════════════════════════
#  Параметры (метаданные)
#  (attr, vtype)
#  vtype: "int" / "float" / "ms" (в GUI миллисекунды, в config секунды) / "pct" (0..1 в config)
# ═══════════════════════════════════════════════════════════
SOURCE_PARAM_META = [
    ("BITE_FORCE_HOOK", "float"),
    ("FISH_GAME_SIZE", "int"),
    ("DEAD_ZONE", "int"),
    ("HOLD_MIN_S", "ms"),
    ("HOLD_MAX_S", "ms"),
    ("HOLD_GAIN", "float"),
    ("PREDICT_AHEAD", "float"),
    ("SPEED_DAMPING", "float"),
    ("MAX_FISH_BAR_DIST", "int"),
    ("VELOCITY_SMOOTH", "float"),
    ("TRACK_MIN_ANGLE", "float"),
    ("TRACK_MAX_ANGLE", "float"),
    ("REGION_UP", "int"),
    ("REGION_DOWN", "int"),
    ("REGION_X", "int"),
    ("POST_CATCH_DELAY", "float"),
    ("SHAKE_HEAD_TIME", "float"),
    ("INITIAL_PRESS_TIME", "float"),
    ("VERIFY_CONSECUTIVE", "int"),
    ("SUCCESS_PROGRESS", "pct"),
]

CUSTOM_PARAM_META = [
    ("SHAKE_HEAD_GAP", "float"),
    ("SHAKE_HEAD_RESET_REPEAT", "int"),
    ("SHAKE_HEAD_RESET_INTERVAL", "float"),
    ("FAST_LOCK_JUMP_PX", "int"),
    ("FAST_LOCK_SPEED_PX_S", "float"),
    ("FAST_LOCK_LOOKAHEAD_S", "ms"),
    ("FAST_LOCK_TRIGGER_ERR", "pct"),
    ("FAST_LOCK_BOOST_GAIN", "float"),
    ("FAST_LOCK_BOOST_MAX_S", "ms"),
    ("FAST_LOCK_DROP_ERR", "pct"),
]


# ═══════════════════════════════════════════════════════════
#  Локализация
# ═══════════════════════════════════════════════════════════
LANG_OPTIONS = [
    ("ru", "Русский"),
    ("en", "English"),
]


def _lang_name(lang: str) -> str:
    for k, label in LANG_OPTIONS:
        if k == lang:
            return label
    return lang


I18N = {
    "ru": {
        # окна/секции
        "title": "VRC авто-рыбалка 263302",
        "frame_status": " Статус ",
        "frame_yolo": " YOLO (нейросеть) ",
        "frame_log": " Журнал ",
        "frame_params": " Настройки мини-игры (применяются сразу) ",

        # подписи статуса
        "lbl_run_state": "Статус:",
        "lbl_vrchat_window": "Окно VRChat:",
        "lbl_fish_count": "Попыток:",
        "lbl_debug_mode": "Отладка:",

        # кнопки
        "btn_start": "▶ Старт (F9)",
        "btn_stop": "■ Стоп (F10)",
        "btn_debug": "Отладка (F11)",
        "btn_connect": "🔗 Найти окно",
        "btn_screenshot": "📸 Скриншот",
        "btn_clearlog": "🗑 Очистить",
        "btn_whitelist": "🐟 Фильтр рыб",
        "btn_roi": "📐 Выбрать зону",
        "btn_clear_roi": "✕ Сбросить зону",

        # переключатели
        "chk_topmost": "Поверх окон",
        "chk_show_debug": "Окно отладки",
        "chk_fast_lock": "Быстрый захват",

        # язык
        "lbl_lang": "Язык:",

        # yolo
        "yolo_enabled": "YOLO включён",
        "chk_yolo_collect": "Сбор скринов",
        "lbl_device": "Устройство:",
        "yolo_dev_auto": "Авто (GPU если есть)",
        "yolo_dev_cpu": "CPU (процессор)",
        "yolo_dev_gpu": "GPU (нужна CUDA)",
        "yolo_model_ok": "Модель ✓",
        "yolo_model_bad": "Модель ✗ (не найдена)",
        "yolo_train": "Обуч.",
        "yolo_unlabeled": "Неразм.",

        # roi
        "lbl_roi": "Зона поиска:",
        "roi_not_set": "Не задана (весь экран)",
        "roi_select_prompt": "[Зона] Выделите мышью область мини-игры. Enter — подтвердить, Esc — отмена",
        "roi_window_name": "Выделите зону рыбалки — Enter=ОК / Esc=Отмена",
        "roi_set_ok": "[Зона] ✓ Установлена: X={x} Y={y} {w}×{h}",
        "roi_cancel": "[Зона] Отменено (слишком маленькая или нажат Esc)",
        "roi_cleared": "[Зона] Сброшена — поиск по всему экрану",

        # параметры
        "group_source": "Основные настройки",
        "group_custom": "Расширенные настройки",
        "btn_apply_params": "Применить",
        "btn_reset_params": "По умолчанию",
        "btn_preset_aggr": "Пресет: агрессивный",
        "btn_preset_bal": "Пресет: сбалансированный",

        # whitelist
        "wl_title": "Фильтр рыб",
        "wl_hint": "Отметьте, какую рыбу ловить:",
        "wl_apply": "Применить",
        "wl_updated": "[Фильтр] Обновлён: {items}",
        "fish_black": "Чёрная",
        "fish_white": "Белая",
        "fish_copper": "Медная",
        "fish_green": "Зелёная",
        "fish_blue": "Синяя",
        "fish_purple": "Фиолетовая",
        "fish_pink": "Розовая",
        "fish_red": "Красная",
        "fish_rainbow": "Радужная",

        # общие слова
        "on": "Вкл",
        "off": "Выкл",
        "ready": "Готов",
        "not_connected": "Не подключено",
        "not_found": "Не найдено",

        # сообщения лога
        "err_non_ascii": "[Ошибка] В пути к программе есть кириллица или спецсимволы — модели и картинки могут не загрузиться!",
        "cur_path": "  Путь: {path}",
        "move_to_ascii": "  Переместите папку на путь без кириллицы, например: D:\\fish",
        "err_vrchat_not_found": "[Ошибка] Окно VRChat не найдено! Запустите игру и попробуйте снова.",
        "sys_started": "[Система] ▶ Авто-рыбалка запущена",
        "sys_stopped": "[Система] ■ Остановлено",
        "sys_connected": "[Система] Подключено: {title}",
        "err_connect": "[Ошибка] Окно VRChat не найдено",
        "err_screenshot_not_connected": "[Ошибка] Сначала подключите окно VRChat",
        "screenshot_saved": "[Скрин] Сохранён ({w}×{h}) → debug/manual_screenshot.png",
        "window_region": "        Окно: x={x} y={y} {w}×{h}",
        "err_screenshot_failed": "[Ошибка] Не удалось сделать скриншот",
        "debug_state": "[Система] Отладка: {state}",
        "debug_hint": "[Подсказка] Скриншоты отладки сохраняются в debug/, детектор показывает уверенность",
        "debug_window_state": "[Отладка] Окно отладки: {state}",
        "fast_lock_state": "[Управление] Быстрый захват: {state}",
        "yolo_collect_on": "[YOLO] Сбор данных включён — скриншоты будут сохраняться во время рыбалки",
        "yolo_collect_off": "[YOLO] Сбор данных выключен",
        "yolo_dev_changed": "[YOLO] Устройство: {device} — применится после перезапуска",
        "err_select_roi_connect": "[Ошибка] Сначала подключите окно VRChat (кнопка «Найти окно»)",
        "err_select_roi_screenshot": "[Ошибка] Не удалось сделать скриншот — выбор зоны невозможен",
        "roi_fullscreen": "[Зона] Сброшена — поиск по всему экрану",
        "params_updated": "[Настройки] Сохранено: {changes}",
        "params_reset": "[Настройки] Сброшено к значениям по умолчанию",
        "preset_applied": "[Настройки] Применён пресет: {name}",
        "warn_save_settings": "[Внимание] Не удалось сохранить настройки: {err}",
        "warn_load_settings": "[Внимание] Не удалось загрузить настройки: {err}",
        "err_capture_exception": "[Ошибка] Сбой захвата экрана: {err}",
        "log_saved": "[Система] Журнал сохранён → {path}",
        "github": "GitHub: https://github.com/day123123123/vrc-auto-fish",
    },

    "en": {
        "title": "VRC auto fish 263302",
        "frame_status": " Status ",
        "frame_yolo": " YOLO detection ",
        "frame_log": " Log ",
        "frame_params": " Mini-game params (live) ",

        "lbl_run_state": "State:",
        "lbl_vrchat_window": "VRChat window:",
        "lbl_fish_count": "Fishing count:",
        "lbl_debug_mode": "Debug:",

        "btn_start": "▶ Start (F9)",
        "btn_stop": "■ Stop (F10)",
        "btn_debug": "Debug (F11)",
        "btn_connect": "🔗 Connect window",
        "btn_screenshot": "📸 Save screenshot",
        "btn_clearlog": "🗑 Clear log",
        "btn_whitelist": "🐟 Whitelist",
        "btn_roi": "📐 Select ROI",
        "btn_clear_roi": "✕ Clear ROI",

        "chk_topmost": "Always on top",
        "chk_show_debug": "Debug window",
        "chk_fast_lock": "Fast lock",

        "lbl_lang": "Lang:",

        "yolo_enabled": "YOLO enabled",
        "chk_yolo_collect": "Collect data",
        "lbl_device": "Device:",
        "yolo_dev_auto": "Auto (prefer GPU)",
        "yolo_dev_cpu": "CPU",
        "yolo_dev_gpu": "GPU (CUDA required)",
        "yolo_model_ok": "Model ✓",
        "yolo_model_bad": "Model ✗",
        "yolo_train": "Train",
        "yolo_unlabeled": "Unlabeled",

        "lbl_roi": "ROI:",
        "roi_not_set": "Not set (fullscreen search)",
        "roi_select_prompt": "[ROI] Select fishing UI area in the popup window. Enter=OK, Esc=Cancel",
        "roi_window_name": "Select Fishing ROI - Enter=OK / Esc=Cancel",
        "roi_set_ok": "[ROI] ✓ ROI set: X={x} Y={y} {w}x{h}",
        "roi_cancel": "[ROI] Cancelled (too small or Esc)",
        "roi_cleared": "[ROI] Cleared — fullscreen search will be used",

        "group_source": "Source params",
        "group_custom": "Custom params",
        "btn_apply_params": "Apply",
        "btn_reset_params": "Reset",
        "btn_preset_aggr": "Fast lock (hard)",
        "btn_preset_bal": "Fast lock (balanced)",

        "wl_title": "Fish whitelist",
        "wl_hint": "Select fish to catch:",
        "wl_apply": "Apply",
        "wl_updated": "[Whitelist] Updated: {items}",
        "fish_black": "Black",
        "fish_white": "White",
        "fish_copper": "Copper",
        "fish_green": "Green",
        "fish_blue": "Blue",
        "fish_purple": "Purple",
        "fish_pink": "Pink",
        "fish_red": "Red",
        "fish_rainbow": "Rainbow",

        "on": "On",
        "off": "Off",
        "ready": "Ready",
        "not_connected": "Not connected",
        "not_found": "Not found",

        "err_non_ascii": "[Error] Program path contains non-ASCII characters; this may break image/model loading!",
        "cur_path": "  Current path: {path}",
        "move_to_ascii": "  Move the program to an ASCII-only path, e.g.: D:\\fish",
        "err_vrchat_not_found": "[Error] VRChat window not found. Make sure the game is running.",
        "sys_started": "[System] ▶ Auto fishing started",
        "sys_stopped": "[System] ■ Stopped",
        "sys_connected": "[System] Connected: {title}",
        "err_connect": "[Error] VRChat window not found",
        "err_screenshot_not_connected": "[Error] Screenshot failed: VRChat window not connected",
        "screenshot_saved": "[Screenshot] Saved ({w}×{h}) → debug/manual_screenshot.png",
        "window_region": "            Window region: x={x} y={y} w={w} h={h}",
        "err_screenshot_failed": "[Error] Screenshot failed",
        "debug_state": "[System] Debug: {state}",
        "debug_hint": "[Hint] Debug screenshots will be saved to debug/, detector prints confidence",
        "debug_window_state": "[Debug] Debug window: {state}",
        "fast_lock_state": "[Control] Fast lock: {state}",
        "yolo_collect_on": "[YOLO] Data collection enabled — screenshots will be saved while fishing",
        "yolo_collect_off": "[YOLO] Data collection disabled",
        "yolo_dev_changed": "[YOLO] Device: {device} — effective after restart",
        "err_select_roi_connect": "[Error] Please connect to VRChat window first",
        "err_select_roi_screenshot": "[Error] Screenshot failed — cannot select ROI",
        "roi_fullscreen": "[ROI] Cleared — fullscreen search will be used",
        "params_updated": "[Params] Updated & saved: {changes}",
        "params_reset": "[Params] Reset to defaults",
        "preset_applied": "[Params] Preset applied: {name}",
        "warn_save_settings": "[Warning] Failed to save settings: {err}",
        "warn_load_settings": "[Warning] Failed to load settings: {err}",
        "err_capture_exception": "[Error] Capture exception: {err}",
        "log_saved": "[System] Log saved → {path}",
        "github": "GitHub: https://github.com/day123123123/vrc-auto-fish",
    },
}


def tr(lang: str, key: str, **kwargs) -> str:
    """Перевод с fallback: lang → en → ru → key."""
    lang = lang if lang in I18N else "en"
    s = I18N.get(lang, {}).get(key)
    if s is None:
        s = I18N.get("en", {}).get(key)
    if s is None:
        s = I18N.get("ru", {}).get(key)
    if s is None:
        s = key
    try:
        return s.format(**kwargs)
    except Exception:
        return s


# ═══════════════════════════════════════════════════════════
#  Параметры: подписи + подсказки по языкам
#  Храним отдельно, чтобы не размазывать по коду
# ═══════════════════════════════════════════════════════════
PARAM_TEXT = {
    # SOURCE
    "BITE_FORCE_HOOK": {
        "ru": ("Автоподсечка (с)", "Через сколько секунд без поклёвки автоматически подсекать"),
        "en": ("Force hook (s)", "If no bite for N seconds, force hook to enter the mini-game"),
    },
    "FISH_GAME_SIZE": {
        "ru": ("Размер рыбы (px)", "Размер иконки рыбы в пикселях (меньше значение → шире поиск)"),
        "en": ("Fish size (px)", "Approx fish icon size (smaller → higher search scale)"),
    },
    "DEAD_ZONE": {
        "ru": ("Мёртвая зона (px)", "Зона нечувствительности: больше = реже зажимает кнопку"),
        "en": ("Dead zone (px)", "Bigger → easier to trigger hold"),
    },
    "HOLD_MIN_S": {
        "ru": ("Мин. удержание (мс)", "Базовое время зажатия. Меньше → быстрее падает, больше → дольше висит"),
        "en": ("Anti-gravity base (ms)", "Lower → falls faster, higher → more floating"),
    },
    "HOLD_MAX_S": {
        "ru": ("Макс. удержание (мс)", "Максимальное время одного зажатия кнопки"),
        "en": ("Max hold (ms)", "Maximum duration of a single hold"),
    },
    "HOLD_GAIN": {
        "ru": ("Сила удержания", "Множитель: ошибка позиции × сила = доп. время зажатия"),
        "en": ("Hold gain", "Position error × gain = extra hold time"),
    },
    "PREDICT_AHEAD": {
        "ru": ("Упреждение (с)", "На сколько секунд вперёд предсказывать позицию полоски"),
        "en": ("Predict ahead (s)", "Predict position this far into the future"),
    },
    "SPEED_DAMPING": {
        "ru": ("Гашение скорости", "Падает быстро → зажимать дольше, поднимается → зажимать меньше"),
        "en": ("Speed damping", "Falling fast → hold more; rising fast → hold less"),
    },
    "MAX_FISH_BAR_DIST": {
        "ru": ("Макс. расстояние (px)", "Если рыба и полоска дальше этого — считать ложным срабатыванием"),
        "en": ("Max distance (px)", "If fish-bar distance exceeds this, treat as false detection"),
    },
    "VELOCITY_SMOOTH": {
        "ru": ("Сглаживание (0–1)", "Чем больше, тем плавнее реакция на изменение скорости"),
        "en": ("Velocity smooth", "0~1, higher → smoother"),
    },
    "TRACK_MIN_ANGLE": {
        "ru": ("Порог наклона (°)", "Если трек наклонён сильнее — включается автоповорот"),
        "en": ("Rotation threshold (°)", "Enable rotation when track tilt exceeds this"),
    },
    "TRACK_MAX_ANGLE": {
        "ru": ("Макс. наклон (°)", "Наклон больше этого — игнорировать (скорее всего ошибка)"),
        "en": ("Max angle (°)", "Above this angle treat as false detection"),
    },
    "REGION_UP": {
        "ru": ("Поиск вверх (px)", "Сколько пикселей вверх от полоски искать рыбу"),
        "en": ("Search up (px)", "Search N pixels above the locked white bar"),
    },
    "REGION_DOWN": {
        "ru": ("Поиск вниз (px)", "Сколько пикселей вниз от полоски искать рыбу"),
        "en": ("Search down (px)", "Search N pixels below the locked white bar"),
    },
    "REGION_X": {
        "ru": ("Поиск по X (px)", "Ширина зоны поиска: ±N пикселей от центра полоски"),
        "en": ("Search X (px)", "Search within ±N pixels around white bar center"),
    },
    "POST_CATCH_DELAY": {
        "ru": ("Пауза после вылова (с)", "Ожидание перед следующим забросом после успеха или провала"),
        "en": ("Post-catch delay (s)", "Wait N seconds after success/fail before casting again"),
    },
    "SHAKE_HEAD_TIME": {
        "ru": ("Время кивка (с)", "Длительность движения головой для заброса. 0 = выключить"),
        "en": ("Shake time (s)", "Hold duration per shake segment, 0=disable"),
    },
    "INITIAL_PRESS_TIME": {
        "ru": ("Нажатие в начале (с)", "Как долго зажимать кнопку при старте мини-игры"),
        "en": ("Initial press (s)", "Press duration at start (0.5s fixed start delay)"),
    },
    "VERIFY_CONSECUTIVE": {
        "ru": ("Кадров подтверждения", "Сколько кадров подряд нужно увидеть UI, чтобы начать игру"),
        "en": ("Verify frames", "How many consecutive frames to confirm mini-game start"),
    },
    "SUCCESS_PROGRESS": {
        "ru": ("Порог успеха (%)", "Прогресс выше этого значения = рыба поймана"),
        "en": ("Success threshold (%)", "If progress exceeds this percentage, treat as success"),
    },

    # CUSTOM
    "SHAKE_HEAD_GAP": {
        "ru": ("Пауза кивка (с)", "Задержка между движениями головой"),
        "en": ("Shake gap (s)", "Delay between two shake segments"),
    },
    "SHAKE_HEAD_RESET_REPEAT": {
        "ru": ("Повторов сброса", "Сколько раз отправлять Left/Right=0 до и после кивка"),
        "en": ("Reset repeat (times)", "How many times to send Left/Right=0 around shake"),
    },
    "SHAKE_HEAD_RESET_INTERVAL": {
        "ru": ("Интервал сброса (с)", "Пауза между повторами сброса"),
        "en": ("Reset interval (s)", "Interval between reset repeats"),
    },
    "FAST_LOCK_JUMP_PX": {
        "ru": ("Прыжок (px)", "Если рыба прыгнула за кадр больше — ускорить реакцию"),
        "en": ("Fast lock jump (px)", "If fish jumps more than this in one frame, reduce smoothing lag"),
    },
    "FAST_LOCK_SPEED_PX_S": {
        "ru": ("Скорость (px/с)", "При скорости рыбы выше этой — включить быстрый режим"),
        "en": ("Fast lock speed (px/s)", "Enable fast lock when fish speed exceeds this"),
    },
    "FAST_LOCK_LOOKAHEAD_S": {
        "ru": ("Упреждение (мс)", "На сколько мс вперёд предсказывать в быстром режиме"),
        "en": ("Fast lock lookahead (ms)", "Prediction lookahead used in fast lock"),
    },
    "FAST_LOCK_TRIGGER_ERR": {
        "ru": ("Порог включения (%)", "Отклонение от центра для активации быстрого режима"),
        "en": ("Trigger threshold (%)", "Enable fast lock when error exceeds this"),
    },
    "FAST_LOCK_BOOST_GAIN": {
        "ru": ("Доп. усиление", "Дополнительная сила удержания в быстром режиме"),
        "en": ("Fast lock boost gain", "Extra hold gain during fast lock"),
    },
    "FAST_LOCK_BOOST_MAX_S": {
        "ru": ("Макс. доп. зажатие (мс)", "Максимум дополнительного времени зажатия в быстром режиме"),
        "en": ("Max boost hold (ms)", "Maximum extra hold time during fast lock"),
    },
    "FAST_LOCK_DROP_ERR": {
        "ru": ("Порог отпускания (%)", "Рыба ниже полоски на столько % — быстро отпустить"),
        "en": ("Drop threshold (%)", "If fish is clearly below the bar, release faster"),
    },
}


FAST_LOCK_PRESETS = {
    "aggressive": {
        "FAST_LOCK_ENABLED": True,
        "FAST_LOCK_JUMP_PX": 12,
        "FAST_LOCK_SPEED_PX_S": 520.0,
        "FAST_LOCK_LOOKAHEAD_S": 0.055,
        "FAST_LOCK_TRIGGER_ERR": 0.10,
        "FAST_LOCK_BOOST_GAIN": 0.070,
        "FAST_LOCK_BOOST_MAX_S": 0.085,
        "FAST_LOCK_DROP_ERR": 0.22,
        "HOLD_GAIN": 0.048,
        "HOLD_MAX_S": 0.120,
    },
    "balanced": {
        "FAST_LOCK_ENABLED": True,
        "FAST_LOCK_JUMP_PX": 16,
        "FAST_LOCK_SPEED_PX_S": 700.0,
        "FAST_LOCK_LOOKAHEAD_S": 0.040,
        "FAST_LOCK_TRIGGER_ERR": 0.15,
        "FAST_LOCK_BOOST_GAIN": 0.050,
        "FAST_LOCK_BOOST_MAX_S": 0.060,
        "FAST_LOCK_DROP_ERR": 0.28,
        "HOLD_GAIN": 0.040,
        "HOLD_MAX_S": 0.100,
    },
}


class FishingApp:
    """VRChat auto-fishing — main window"""

    def __init__(self, root: tk.Tk):
        self.root = root

        # текущий язык UI
        self.ui_lang = "ru"

        # ── Экземпляр бота ──
        self.bot = FishingBot()
        self.bot_thread: threading.Thread | None = None

        # ── Переменные параметров ──
        self._param_vars: dict[str, tuple[tk.StringVar, str]] = {}

        # ── Построение UI ──
        self._build_ui()

        # ── Загрузка настроек ──
        self._load_settings()

        # применяем язык (на случай если он загрузился из settings)
        self._apply_language()

        # ── Предзагрузка YOLO ──
        if self.bot.yolo is None:
            self._preload_yolo()

        # ── Глобальные хоткеи ──
        keyboard.add_hotkey(config.HOTKEY_TOGGLE, self._toggle_from_hotkey)
        keyboard.add_hotkey(config.HOTKEY_STOP, self._stop_from_hotkey)
        keyboard.add_hotkey(config.HOTKEY_DEBUG, self._toggle_debug_from_hotkey)

        # ── Опрос ──
        self._poll()

        # ── Закрытие ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._log_msg(self._t("github"))

    # ══════════════════════════════════════════════════════
    #  i18n helpers
    # ══════════════════════════════════════════════════════

    def _t(self, key: str, **kwargs) -> str:
        return tr(self.ui_lang, key, **kwargs)

    def _param_text(self, attr: str) -> tuple[str, str]:
        data = PARAM_TEXT.get(attr, {})
        label, tip = data.get(self.ui_lang) or data.get("en") or data.get("ru") or (attr, "")
        return label, tip

    def _translate_state(self, state: str) -> str:
        """Display bot.state in the selected UI language. Does not modify core.bot."""
        if state is None:
            return ""
        s = str(state)

        # en → canonical
        en_to_key = {
            "ready": "ready",
            "running": "running",
            "stopped": "stopped",
            "stop": "stopped",
        }

        key = en_to_key.get(s.lower())

        if key == "ready":
            return self._t("ready")
        if key == "running":
            if self.ui_lang == "ru":
                return "Работает"
            return "Running"
        if key == "stopped":
            if self.ui_lang == "ru":
                return "Остановлено"
            return "Stopped"

        # unknown — show as-is
        return s

    # ══════════════════════════════════════════════════════
    #  Построение UI
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self.root.title(self._t("title"))
        self.root.geometry("580x800")
        self.root.resizable(True, True)
        self.root.minsize(520, 600)
        self.root.attributes("-topmost", False)

        pad = {"padx": 10, "pady": 5}

        # ── Статус ──
        self.frm_status = ttk.LabelFrame(self.root, text=self._t("frame_status"))
        self.frm_status.pack(fill="x", **pad)

        grid_pad = {"padx": 8, "pady": 3, "sticky": "w"}

        self.var_state = tk.StringVar(value=self._t("ready"))
        self.var_window = tk.StringVar(value=self._t("not_connected"))
        self.var_count = tk.StringVar(value="0")
        self.var_debug = tk.StringVar(value=self._t("off"))

        self.lbl_cap_state = ttk.Label(self.frm_status, text=self._t("lbl_run_state"))
        self.lbl_cap_state.grid(row=0, column=0, **grid_pad)
        self.lbl_state = ttk.Label(self.frm_status, textvariable=self.var_state, foreground="gray")
        self.lbl_state.grid(row=0, column=1, **grid_pad)

        self.lbl_cap_window = ttk.Label(self.frm_status, text=self._t("lbl_vrchat_window"))
        self.lbl_cap_window.grid(row=1, column=0, **grid_pad)
        self.lbl_window = ttk.Label(self.frm_status, textvariable=self.var_window)
        self.lbl_window.grid(row=1, column=1, **grid_pad)

        self.lbl_cap_count = ttk.Label(self.frm_status, text=self._t("lbl_fish_count"))
        self.lbl_cap_count.grid(row=2, column=0, **grid_pad)
        self.lbl_count = ttk.Label(self.frm_status, textvariable=self.var_count)
        self.lbl_count.grid(row=2, column=1, **grid_pad)

        self.lbl_cap_debug = ttk.Label(self.frm_status, text=self._t("lbl_debug_mode"))
        self.lbl_cap_debug.grid(row=3, column=0, **grid_pad)
        self.lbl_debug = ttk.Label(self.frm_status, textvariable=self.var_debug)
        self.lbl_debug.grid(row=3, column=1, **grid_pad)

        # ── Управление ──
        self.frm_ctrl = ttk.Frame(self.root)
        self.frm_ctrl.pack(fill="x", **pad)

        self.btn_start = ttk.Button(self.frm_ctrl, text=self._t("btn_start"), command=self._on_start, width=15)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(self.frm_ctrl, text=self._t("btn_stop"), command=self._on_stop, width=15, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.btn_debug = ttk.Button(self.frm_ctrl, text=self._t("btn_debug"), command=self._on_toggle_debug, width=15)
        self.btn_debug.pack(side="left", padx=5)

        # ── Вспом. кнопки ──
        self.frm_aux = ttk.Frame(self.root)
        self.frm_aux.pack(fill="x", **pad)

        self.btn_connect = ttk.Button(self.frm_aux, text=self._t("btn_connect"), command=self._on_connect, width=15)
        self.btn_connect.pack(side="left", padx=5)

        self.btn_screenshot = ttk.Button(self.frm_aux, text=self._t("btn_screenshot"), command=self._on_screenshot, width=15)
        self.btn_screenshot.pack(side="left", padx=5)

        self.btn_clearlog = ttk.Button(self.frm_aux, text=self._t("btn_clearlog"), command=self._on_clear_log, width=12)
        self.btn_clearlog.pack(side="left", padx=5)

        self.btn_whitelist = ttk.Button(self.frm_aux, text=self._t("btn_whitelist"), command=self._on_whitelist, width=12)
        self.btn_whitelist.pack(side="left", padx=5)

        # ── Переключатели + язык ──
        self.frm_toggles = ttk.Frame(self.root)
        self.frm_toggles.pack(fill="x", **pad)

        self.var_topmost = tk.BooleanVar(value=False)
        self.chk_topmost = ttk.Checkbutton(self.frm_toggles, text=self._t("chk_topmost"), variable=self.var_topmost, command=self._on_topmost)
        self.chk_topmost.pack(side="left", padx=5)

        self.var_show_debug = tk.BooleanVar(value=config.SHOW_DEBUG)
        self.chk_show_debug = ttk.Checkbutton(self.frm_toggles, text=self._t("chk_show_debug"), variable=self.var_show_debug, command=self._on_debug_toggle)
        self.chk_show_debug.pack(side="left", padx=5)

        self.var_fast_lock = tk.BooleanVar(value=config.FAST_LOCK_ENABLED)
        self.chk_fast_lock = ttk.Checkbutton(self.frm_toggles, text=self._t("chk_fast_lock"), variable=self.var_fast_lock, command=self._on_fast_lock_toggle)
        self.chk_fast_lock.pack(side="left", padx=5)

        # селектор языка
        self.lbl_lang = ttk.Label(self.frm_toggles, text=self._t("lbl_lang"))
        self.lbl_lang.pack(side="left", padx=(12, 2))

        self.var_ui_lang = tk.StringVar(value=self.ui_lang)
        self.cmb_lang = ttk.Combobox(self.frm_toggles, textvariable=self.var_ui_lang,
                                    values=[k for k, _ in LANG_OPTIONS], state="readonly", width=5)
        self.cmb_lang.pack(side="left", padx=2)
        self.cmb_lang.bind("<<ComboboxSelected>>", self._on_lang_change)

        # ── YOLO ──
        self.frm_yolo = ttk.LabelFrame(self.root, text=self._t("frame_yolo"))
        self.frm_yolo.pack(fill="x", **pad)

        config.USE_YOLO = True
        self.lbl_yolo_enabled = ttk.Label(self.frm_yolo, text=self._t("yolo_enabled"))
        self.lbl_yolo_enabled.pack(side="left", padx=5)

        self.var_yolo_collect = tk.BooleanVar(value=config.YOLO_COLLECT)
        self.chk_yolo_collect = ttk.Checkbutton(self.frm_yolo, text=self._t("chk_yolo_collect"),
                                                variable=self.var_yolo_collect, command=self._on_yolo_collect_toggle)
        self.chk_yolo_collect.pack(side="left", padx=5)

        self.lbl_yolo_device = ttk.Label(self.frm_yolo, text=self._t("lbl_device"))
        self.lbl_yolo_device.pack(side="left", padx=(10, 2))

        self.var_yolo_device = tk.StringVar(value=config.YOLO_DEVICE)
        self.cmb_yolo_dev = ttk.Combobox(self.frm_yolo, textvariable=self.var_yolo_device,
                                         values=["auto", "cpu", "gpu"], state="readonly", width=5)
        self.cmb_yolo_dev.pack(side="left", padx=2)
        self.cmb_yolo_dev.bind("<<ComboboxSelected>>", self._on_yolo_device_change)

        self.var_yolo_status = tk.StringVar(value="")
        self._update_yolo_status()
        self.lbl_yolo_status = ttk.Label(self.frm_yolo, textvariable=self.var_yolo_status, foreground="gray")
        self.lbl_yolo_status.pack(side="left", padx=10)

        # ── ROI ──
        self.frm_roi = ttk.Frame(self.root)
        self.frm_roi.pack(fill="x", **pad)

        self.btn_roi = ttk.Button(self.frm_roi, text=self._t("btn_roi"), command=self._on_select_roi, width=15)
        self.btn_roi.pack(side="left", padx=5)

        self.btn_clear_roi = ttk.Button(self.frm_roi, text=self._t("btn_clear_roi"), command=self._on_clear_roi, width=12)
        self.btn_clear_roi.pack(side="left", padx=5)

        self.var_roi = tk.StringVar(value=self._t("roi_not_set"))
        self.lbl_cap_roi = ttk.Label(self.frm_roi, text=self._t("lbl_roi"))
        self.lbl_cap_roi.pack(side="left", padx=(10, 2))
        self.lbl_roi = ttk.Label(self.frm_roi, textvariable=self.var_roi, foreground="gray")
        self.lbl_roi.pack(side="left")

        # ── Параметры (в отдельном контейнере, чтобы пересобирать при смене языка) ──
        self.frm_params_container = ttk.Frame(self.root)
        self.frm_params_container.pack(fill="x", **pad)
        self._build_params_panel()

        # ── Лог ──
        self.frm_log = ttk.LabelFrame(self.root, text=self._t("frame_log"))
        self.frm_log.pack(fill="both", expand=True, **pad)

        self.txt_log = scrolledtext.ScrolledText(
            self.frm_log, height=14, state="disabled",
            font=("Consolas", 9), wrap="word",
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)

    def _apply_language(self):
        """Обновить все тексты UI под текущий язык."""
        # обновляем заголовок
        self.root.title(self._t("title"))

        # рамки
        self.frm_status.config(text=self._t("frame_status"))
        self.frm_yolo.config(text=self._t("frame_yolo"))
        self.frm_log.config(text=self._t("frame_log"))

        # статус-подписи
        self.lbl_cap_state.config(text=self._t("lbl_run_state"))
        self.lbl_cap_window.config(text=self._t("lbl_vrchat_window"))
        self.lbl_cap_count.config(text=self._t("lbl_fish_count"))
        self.lbl_cap_debug.config(text=self._t("lbl_debug_mode"))

        # кнопки
        self.btn_start.config(text=self._t("btn_start"))
        self.btn_stop.config(text=self._t("btn_stop"))
        self.btn_debug.config(text=self._t("btn_debug"))
        self.btn_connect.config(text=self._t("btn_connect"))
        self.btn_screenshot.config(text=self._t("btn_screenshot"))
        self.btn_clearlog.config(text=self._t("btn_clearlog"))
        self.btn_whitelist.config(text=self._t("btn_whitelist"))
        self.btn_roi.config(text=self._t("btn_roi"))
        self.btn_clear_roi.config(text=self._t("btn_clear_roi"))

        # переключатели
        self.chk_topmost.config(text=self._t("chk_topmost"))
        self.chk_show_debug.config(text=self._t("chk_show_debug"))
        self.chk_fast_lock.config(text=self._t("chk_fast_lock"))

        # язык
        self.lbl_lang.config(text=self._t("lbl_lang"))

        # yolo
        self.lbl_yolo_enabled.config(text=self._t("yolo_enabled"))
        self.chk_yolo_collect.config(text=self._t("chk_yolo_collect"))
        self.lbl_yolo_device.config(text=self._t("lbl_device"))
        self._update_yolo_status()

        # roi
        self.lbl_cap_roi.config(text=self._t("lbl_roi"))
        # если ROI не задан, обновляем текст; если задан — оставляем координаты
        if config.DETECT_ROI is None:
            self.var_roi.set(self._t("roi_not_set"))
            self.lbl_roi.config(foreground="gray")

        # debug/status тексты-переменные (переустановим мягко)
        self.var_debug.set(self._t("on") if getattr(self.bot, "debug_mode", False) else self._t("off"))

        # пересобираем панель параметров
        self._rebuild_params_panel()

    def _on_lang_change(self, _event=None):
        self.ui_lang = self.var_ui_lang.get() or "en"
        self._save_settings()
        self._apply_language()

    # ══════════════════════════════════════════════════════
    #  Панель параметров
    # ══════════════════════════════════════════════════════

    def _build_params_panel(self):
        """Построить панель параметров (в контейнере)."""
        # очистим контейнер
        for w in self.frm_params_container.winfo_children():
            w.destroy()

        self._param_vars = {}

        self.frm_params = ttk.LabelFrame(self.frm_params_container, text=self._t("frame_params"))
        self.frm_params.pack(fill="x")

        row = 0
        row = self._render_param_group(self.frm_params, SOURCE_PARAM_META, row, self._t("group_source"))
        ttk.Separator(self.frm_params, orient="horizontal").grid(
            row=row, column=0, columnspan=6, sticky="ew", padx=6, pady=(4, 2)
        )
        row += 1
        row = self._render_param_group(self.frm_params, CUSTOM_PARAM_META, row, self._t("group_custom"))

        # кнопки
        btn_frame = ttk.Frame(self.frm_params)
        btn_frame.grid(row=row, column=0, columnspan=6, pady=(5, 5), sticky="e", padx=10)

        self.btn_apply_params = ttk.Button(btn_frame, text=self._t("btn_apply_params"), command=self._apply_params, width=10)
        self.btn_apply_params.pack(side="left", padx=3)

        self.btn_reset_params = ttk.Button(btn_frame, text=self._t("btn_reset_params"), command=self._reset_params, width=10)
        self.btn_reset_params.pack(side="left", padx=3)

        self.btn_preset_aggr = ttk.Button(
            btn_frame, text=self._t("btn_preset_aggr"),
            command=lambda: self._apply_fast_lock_preset("aggressive"),
            width=20 if self.ui_lang in ("ru", "en") else 10
        )
        self.btn_preset_aggr.pack(side="left", padx=3)

        self.btn_preset_bal = ttk.Button(
            btn_frame, text=self._t("btn_preset_bal"),
            command=lambda: self._apply_fast_lock_preset("balanced"),
            width=20 if self.ui_lang in ("ru", "en") else 10
        )
        self.btn_preset_bal.pack(side="left", padx=3)

    def _rebuild_params_panel(self):
        """Пересобрать панель параметров без потери config значений."""
        self._build_params_panel()

    def _render_param_group(self, parent, meta, start_row: int, title: str) -> int:
        cols_per_row = 2
        gpad = {"padx": 4, "pady": 2}

        ttk.Label(parent, text=title).grid(
            row=start_row, column=0, columnspan=6, sticky="w", padx=6, pady=(2, 2)
        )

        for i, (attr, vtype) in enumerate(meta):
            row = start_row + 1 + (i // cols_per_row)
            col_base = (i % cols_per_row) * 3

            label, tip = self._param_text(attr)

            # текущее значение
            display_val = self._config_to_display(attr, vtype)
            var = tk.StringVar(value=display_val)
            self._param_vars[attr] = (var, vtype)

            # ширина label зависит от языка
            lbl_width = 22 if self.ui_lang in ("ru", "en") else 12
            lbl = ttk.Label(parent, text=label, width=lbl_width, anchor="e")
            lbl.grid(row=row, column=col_base, sticky="e", **gpad)

            entry = ttk.Entry(parent, textvariable=var, width=8, justify="center")
            entry.grid(row=row, column=col_base + 1, sticky="w", **gpad)

            entry.bind("<Return>", lambda e: self._apply_params())
            entry.bind("<FocusOut>", lambda e: self._apply_params())

            if tip:
                self._create_tooltip(entry, tip)

        rows_used = (len(meta) + cols_per_row - 1) // cols_per_row
        return start_row + 1 + rows_used

    # ══════════════════════════════════════════════════════
    #  Конвертеры значений
    # ══════════════════════════════════════════════════════

    def _config_to_display(self, attr: str, vtype: str) -> str:
        val = getattr(config, attr)
        if vtype == "ms":
            return str(round(val * 1000))
        if vtype == "pct":
            return str(round(val * 100))
        if vtype == "int":
            return str(int(val))
        if vtype == "float":
            if val == 0:
                return "0"
            if abs(val) < 0.001:
                return f"{val:.5f}"
            if abs(val) < 0.1:
                return f"{val:.4f}"
            if abs(val) < 10:
                return f"{val:.3f}"
            return f"{val:.1f}"
        return str(val)

    def _display_to_config(self, text: str, vtype: str):
        text = text.strip()
        if not text:
            return None
        try:
            if vtype == "ms":
                return float(text) / 1000.0
            if vtype == "pct":
                return float(text) / 100.0
            if vtype == "int":
                return int(float(text))
            if vtype == "float":
                return float(text)
        except ValueError:
            return None
        return None

    # ══════════════════════════════════════════════════════
    #  Параметры: применение/сброс
    # ══════════════════════════════════════════════════════

    def _apply_params(self):
        changed = []
        for attr, (var, vtype) in self._param_vars.items():
            new_val = self._display_to_config(var.get(), vtype)
            if new_val is None:
                continue

            old_val = getattr(config, attr)
            if vtype == "ms":
                is_same = abs(old_val - new_val) < 0.0001
            elif vtype == "float":
                is_same = abs(old_val - new_val) < 1e-7
            else:
                is_same = old_val == new_val

            if not is_same:
                setattr(config, attr, new_val)
                changed.append(f"{attr}: {old_val} → {new_val}")

        if changed:
            self._save_settings()
            self._log_msg(self._t("params_updated", changes=", ".join(changed)))

    def _reset_params(self):
        defaults = {
            "BITE_FORCE_HOOK": 18.0,
            "FISH_GAME_SIZE": 20,
            "DEAD_ZONE": 15,
            "HOLD_MIN_S": 0.025,
            "HOLD_MAX_S": 0.100,
            "HOLD_GAIN": 0.040,
            "PREDICT_AHEAD": 0.5,
            "SPEED_DAMPING": 0.00025,
            "MAX_FISH_BAR_DIST": 300,
            "VELOCITY_SMOOTH": 0.5,
            "TRACK_MIN_ANGLE": 3.0,
            "TRACK_MAX_ANGLE": 45.0,
            "REGION_UP": 300,
            "REGION_DOWN": 400,
            "REGION_X": 100,
            "POST_CATCH_DELAY": 3.0,
            "SHAKE_HEAD_TIME": 0.02,
            "SHAKE_HEAD_GAP": 0.05,
            "SHAKE_HEAD_RESET_REPEAT": 2,
            "SHAKE_HEAD_RESET_INTERVAL": 0.01,
            "FAST_LOCK_JUMP_PX": 16,
            "FAST_LOCK_SPEED_PX_S": 700.0,
            "FAST_LOCK_LOOKAHEAD_S": 0.040,
            "FAST_LOCK_TRIGGER_ERR": 0.15,
            "FAST_LOCK_BOOST_GAIN": 0.050,
            "FAST_LOCK_BOOST_MAX_S": 0.060,
            "FAST_LOCK_DROP_ERR": 0.28,
            "INITIAL_PRESS_TIME": 0.2,
            "VERIFY_CONSECUTIVE": 1,
            "SUCCESS_PROGRESS": 0.55,
        }

        for attr, default_val in defaults.items():
            setattr(config, attr, default_val)
            if attr in self._param_vars:
                var, vtype = self._param_vars[attr]
                var.set(self._config_to_display(attr, vtype))

        config.FAST_LOCK_ENABLED = True
        if hasattr(self, "var_fast_lock"):
            self.var_fast_lock.set(config.FAST_LOCK_ENABLED)

        try:
            if os.path.exists(config.SETTINGS_FILE):
                os.remove(config.SETTINGS_FILE)
        except Exception:
            pass

        self._log_msg(self._t("params_reset"))

    def _apply_fast_lock_preset(self, preset_key: str):
        preset = FAST_LOCK_PRESETS.get(preset_key)
        if not preset:
            return

        for attr, val in preset.items():
            setattr(config, attr, val)
            if attr in self._param_vars:
                var, vtype = self._param_vars[attr]
                var.set(self._config_to_display(attr, vtype))

        config.FAST_LOCK_ENABLED = bool(preset.get("FAST_LOCK_ENABLED", True))
        if hasattr(self, "var_fast_lock"):
            self.var_fast_lock.set(config.FAST_LOCK_ENABLED)

        self._save_settings()

        # имя пресета на текущем языке (используем кнопку как источник)
        if preset_key == "aggressive":
            name = self._t("btn_preset_aggr")
        else:
            name = self._t("btn_preset_bal")

        self._log_msg(self._t("preset_applied", name=name))

    # ══════════════════════════════════════════════════════
    #  Сохранение/загрузка
    # ══════════════════════════════════════════════════════

    def _save_settings(self):
        data = {}
        for attr, (_, _vtype) in self._param_vars.items():
            data[attr] = getattr(config, attr)

        data["DETECT_ROI"] = config.DETECT_ROI
        data["YOLO_COLLECT"] = config.YOLO_COLLECT
        data["YOLO_DEVICE"] = config.YOLO_DEVICE
        data["SHOW_DEBUG"] = config.SHOW_DEBUG
        data["FAST_LOCK_ENABLED"] = config.FAST_LOCK_ENABLED
        data["FISH_WHITELIST"] = config.FISH_WHITELIST
        data["UI_LANG"] = self.ui_lang

        try:
            with open(config.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log_msg(self._t("warn_save_settings", err=e))

    def _load_settings(self):
        if not os.path.exists(config.SETTINGS_FILE):
            return
        try:
            with open(config.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # язык
            lang = data.get("UI_LANG")
            if lang in ("ru", "en"):
                self.ui_lang = lang
                if hasattr(self, "var_ui_lang"):
                    self.var_ui_lang.set(lang)

            # небольшая защита от «опасных» значений
            if data.get("HOLD_GAIN", 1) < 0.02:
                data["HOLD_GAIN"] = 0.040
            if data.get("SPEED_DAMPING", 0) > 0.001:
                data["SPEED_DAMPING"] = 0.00025
            if data.get("HOLD_MAX_S", 1) < 0.08:
                data["HOLD_MAX_S"] = 0.100
            if data.get("HOLD_MIN_S", 1) < 0.02:
                data["HOLD_MIN_S"] = 0.025

            for attr, val in data.items():
                if attr == "DETECT_ROI":
                    if val and isinstance(val, list) and len(val) == 4:
                        config.DETECT_ROI = val
                        if hasattr(self, "var_roi"):
                            x, y, w, h = val
                            self.var_roi.set(f"X={x} Y={y} {w}x{h}")
                            self.lbl_roi.config(foreground="green")
                    else:
                        config.DETECT_ROI = None
                elif attr == "YOLO_COLLECT":
                    config.YOLO_COLLECT = bool(val)
                    if hasattr(self, "var_yolo_collect"):
                        self.var_yolo_collect.set(config.YOLO_COLLECT)
                elif attr == "YOLO_DEVICE":
                    if val in ("auto", "cpu", "gpu"):
                        config.YOLO_DEVICE = val
                        if hasattr(self, "var_yolo_device"):
                            self.var_yolo_device.set(val)
                elif attr == "SHOW_DEBUG":
                    config.SHOW_DEBUG = bool(val)
                    if hasattr(self, "var_show_debug"):
                        self.var_show_debug.set(config.SHOW_DEBUG)
                elif attr == "FAST_LOCK_ENABLED":
                    config.FAST_LOCK_ENABLED = bool(val)
                    if hasattr(self, "var_fast_lock"):
                        self.var_fast_lock.set(config.FAST_LOCK_ENABLED)
                elif attr == "FISH_WHITELIST":
                    if isinstance(val, dict):
                        config.FISH_WHITELIST.update(val)
                elif attr == "UI_LANG":
                    pass
                elif attr in self._param_vars:
                    setattr(config, attr, val)
                    var, vtype = self._param_vars[attr]
                    var.set(self._config_to_display(attr, vtype))

        except Exception as e:
            self._log_msg(self._t("warn_load_settings", err=e))

    # ══════════════════════════════════════════════════════
    #  Tooltip
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _create_tooltip(widget, text: str):
        tip_window = [None]

        def show(event):
            if tip_window[0]:
                return
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            lbl = tk.Label(
                tw,
                text=text,
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                font=("", 9),
                padx=4,
                pady=2,
            )
            lbl.pack()
            tip_window[0] = tw

        def hide(_):
            if tip_window[0]:
                tip_window[0].destroy()
                tip_window[0] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ══════════════════════════════════════════════════════
    #  Кнопки/действия
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _has_non_ascii(path: str) -> bool:
        try:
            path.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True

    def _on_start(self):
        if self.bot.running:
            return

        if self._has_non_ascii(config.BASE_DIR):
            self._log_msg(self._t("err_non_ascii"))
            self._log_msg(self._t("cur_path", path=config.BASE_DIR))
            self._log_msg(self._t("move_to_ascii"))
            return

        # connect window
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg(self._t("err_vrchat_not_found"))
                return

        self.var_window.set(f"{self.bot.window.title} (HWND={self.bot.window.hwnd})")

        # apply params once before start
        self._apply_params()

        self.bot.running = True
        # не ломаем оригинальные значения state, но можем поставить «каноническое»
        self.bot.state = "Running"

        if self.bot_thread is None or not self.bot_thread.is_alive():
            self.bot_thread = threading.Thread(target=self.bot.run, daemon=True)
            self.bot_thread.start()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._log_msg(self._t("sys_started"))

    def _on_stop(self):
        self.bot.running = False
        self.bot.input.safe_release()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._log_msg(self._t("sys_stopped"))
        self._save_log()

    def _on_toggle_debug(self):
        self.bot.debug_mode = not self.bot.debug_mode
        state = self._t("on") if self.bot.debug_mode else self._t("off")
        self.var_debug.set(state)
        self._log_msg(self._t("debug_state", state=state))
        if self.bot.debug_mode:
            self._log_msg(self._t("debug_hint"))

    def _on_connect(self):
        if self.bot.window.find():
            self.var_window.set(f"{self.bot.window.title} (HWND={self.bot.window.hwnd})")
            self.bot.screen.reset_capture_method()
            self._log_msg(self._t("sys_connected", title=self.bot.window.title))
        else:
            self.var_window.set(self._t("not_found"))
            self._log_msg(self._t("err_connect"))

    def _on_screenshot(self):
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg(self._t("err_screenshot_not_connected"))
                return

        img, region = self.screen_capture_safe()
        if img is not None:
            self.bot.screen.save_debug(img, "manual_screenshot")
            h, w = img.shape[:2]
            self._log_msg(self._t("screenshot_saved", w=w, h=h))
            if region:
                self._log_msg(self._t("window_region", x=region[0], y=region[1], w=region[2], h=region[3]))
        else:
            self._log_msg(self._t("err_screenshot_failed"))

    def _on_clear_log(self):
        self.txt_log.config(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.config(state="disabled")

    def _on_whitelist(self):
        fish_keys = [
            "fish_black",
            "fish_white",
            "fish_copper",
            "fish_green",
            "fish_blue",
            "fish_purple",
            "fish_pink",
            "fish_red",
            "fish_rainbow",
        ]

        win = tk.Toplevel(self.root)
        win.title(self._t("wl_title"))
        win.geometry("260x360")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=self._t("wl_hint")).pack(pady=(10, 5))

        wl = config.FISH_WHITELIST
        chk_vars: dict[str, tk.BooleanVar] = {}
        for key in fish_keys:
            name = self._t(key)
            var = tk.BooleanVar(value=wl.get(key, True))
            chk_vars[key] = var
            ttk.Checkbutton(win, text=name, variable=var).pack(anchor="w", padx=30)

        def _apply():
            for key, var in chk_vars.items():
                config.FISH_WHITELIST[key] = var.get()
            self._save_settings()
            enabled = [self._t(k) for k in fish_keys if chk_vars[k].get()]
            self._log_msg(self._t("wl_updated", items=", ".join(enabled)))
            win.destroy()

        ttk.Button(win, text=self._t("wl_apply"), command=_apply).pack(pady=10)

    def _on_topmost(self):
        topmost = self.var_topmost.get()
        self.root.wm_attributes("-topmost", 1 if topmost else 0)
        if not topmost:
            self.root.lift()
            self.root.focus_force()

    def _on_debug_toggle(self):
        config.SHOW_DEBUG = self.var_show_debug.get()
        self._save_settings()
        state = self._t("on") if config.SHOW_DEBUG else self._t("off")
        msg_state = state if config.SHOW_DEBUG else (state + " (faster)" if self.ui_lang == "en" else state)
        self._log_msg(self._t("debug_window_state", state=msg_state))
        if not config.SHOW_DEBUG:
            # ★ Безопасное закрытие: через event, а не cv2.destroyWindow из GUI-потока
            # OpenCV GUI-функции нельзя вызывать из разных потоков (deadlock)
            self.bot.stop_debug_window()

    def _on_fast_lock_toggle(self):
        config.FAST_LOCK_ENABLED = self.var_fast_lock.get()
        self._save_settings()
        state = self._t("on") if config.FAST_LOCK_ENABLED else self._t("off")
        self._log_msg(self._t("fast_lock_state", state=state))

    def _preload_yolo(self):
        def _load():
            try:
                from core.bot import _get_yolo_detector
                self.bot.yolo = _get_yolo_detector()
            except Exception as e:
                self._log_msg(f"[YOLO] preload failed: {e}")

        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _on_yolo_collect_toggle(self):
        collect = self.var_yolo_collect.get()
        config.YOLO_COLLECT = collect
        self._save_settings()
        self._log_msg(self._t("yolo_collect_on" if collect else "yolo_collect_off"))
        self._update_yolo_status()

    def _on_yolo_device_change(self, _event=None):
        dev = self.var_yolo_device.get()
        config.YOLO_DEVICE = dev
        self._save_settings()

        labels = {
            "auto": self._t("yolo_dev_auto"),
            "cpu": self._t("yolo_dev_cpu"),
            "gpu": self._t("yolo_dev_gpu"),
        }
        self._log_msg(self._t("yolo_dev_changed", device=labels.get(dev, dev)))

    def _update_yolo_status(self):
        model_ok = os.path.exists(config.YOLO_MODEL)
        unlabeled = os.path.join(config.BASE_DIR, "yolo", "dataset", "images", "unlabeled")
        train = os.path.join(config.BASE_DIR, "yolo", "dataset", "images", "train")

        n_unlabeled = len([f for f in os.listdir(unlabeled) if f.endswith((".png", ".jpg"))]) if os.path.isdir(unlabeled) else 0
        n_train = len([f for f in os.listdir(train) if f.endswith((".png", ".jpg"))]) if os.path.isdir(train) else 0

        parts = []
        parts.append(self._t("yolo_model_ok") if model_ok else self._t("yolo_model_bad"))
        parts.append(f"{self._t('yolo_train')}:{n_train}")
        parts.append(f"{self._t('yolo_unlabeled')}:{n_unlabeled}")
        self.var_yolo_status.set(" | ".join(parts))

    def _on_select_roi(self):
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg(self._t("err_select_roi_connect"))
                return

        img, _ = self.screen_capture_safe()
        if img is None:
            self._log_msg(self._t("err_select_roi_screenshot"))
            return

        self._log_msg(self._t("roi_select_prompt"))

        # ★ Безопасно остановить debug-окно перед открытием ROI-селектора
        # (cv2.selectROI блокирует поток, а debug-поток параллельно зовёт cv2.imshow → deadlock)
        _debug_was_on = config.SHOW_DEBUG
        config.SHOW_DEBUG = False
        self.bot.stop_debug_window()

        try:
            win_name = self._t("roi_window_name")
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            h, w = img.shape[:2]
            dw = min(w, 1280)
            dh = int(h * dw / w)
            cv2.resizeWindow(win_name, dw, dh)

            roi = cv2.selectROI(win_name, img, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(win_name)

            x, y, w_r, h_r = [int(v) for v in roi]
            if w_r > 10 and h_r > 10:
                config.DETECT_ROI = [x, y, w_r, h_r]
                self._save_settings()
                self.var_roi.set(f"X={x} Y={y} {w_r}x{h_r}")
                self.lbl_roi.config(foreground="green")
                self._log_msg(self._t("roi_set_ok", x=x, y=y, w=w_r, h=h_r))
            else:
                self._log_msg(self._t("roi_cancel"))
        finally:
            # ★ Восстановить debug-окно если было включено
            if _debug_was_on:
                config.SHOW_DEBUG = True

    def _on_clear_roi(self):
        config.DETECT_ROI = None
        self._save_settings()
        self.var_roi.set(self._t("roi_not_set"))
        self.lbl_roi.config(foreground="gray")
        self._log_msg(self._t("roi_fullscreen"))

    def screen_capture_safe(self):
        try:
            return self.bot.screen.grab_window(self.bot.window)
        except Exception as e:
            self._log_msg(self._t("err_capture_exception", err=e))
            return None, None

    # ══════════════════════════════════════════════════════
    #  Хоткеи
    # ══════════════════════════════════════════════════════

    def _toggle_from_hotkey(self):
        if self.bot.running:
            self.root.after(0, self._on_stop)
        else:
            self.root.after(0, self._on_start)

    def _stop_from_hotkey(self):
        self.root.after(0, self._on_stop)

    def _toggle_debug_from_hotkey(self):
        self.root.after(0, self._on_toggle_debug)

    # ══════════════════════════════════════════════════════
    #  Poll
    # ══════════════════════════════════════════════════════

    def _poll(self):
        try:
            for _ in range(20):
                msg = log.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass

        # state
        self.var_state.set(self._translate_state(self.bot.state))
        self.var_count.set(str(self.bot.fish_count))

        if self.bot.running:
            self.lbl_state.config(foreground="green")
        else:
            self.lbl_state.config(foreground="gray")

        # thread died unexpectedly
        if self.bot_thread and not self.bot_thread.is_alive() and self.bot.running:
            self.bot.running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

        self.root.after(100, self._poll)

    # ══════════════════════════════════════════════════════
    #  Лог
    # ══════════════════════════════════════════════════════

    def _log_msg(self, msg: str):
        import time
        ts = time.strftime("%H:%M:%S")
        self._append_log(f"[{ts}] {msg}")

    def _append_log(self, text: str):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ══════════════════════════════════════════════════════
    #  Закрытие
    # ══════════════════════════════════════════════════════

    def _on_close(self):
        self.bot.running = False
        self.bot.input.safe_release()
        self._save_settings()
        self._save_log()
        self.root.destroy()

    def _save_log(self):
        path = os.path.join(config.DEBUG_DIR, "last_run.log")
        log.save(path)
        self._log_msg(self._t("log_saved", path=path))
