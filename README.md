# VRChat Fish — Portable

**Портабельная версия** автоматической рыбалки для VRChat (игра [fish!](https://vrchat.com/home/world/wrld_5d5a3e81-069d-4f5d-a962-e8c7bdcf4fbe)).

> **Основан на оригинальном проекте [FoxieBoo/vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)**
> Этот форк добавляет портабельную установку, исправления багов и интерфейс на русском и английском языках.

---

## Отличия от оригинала

| | Оригинал | Этот форк |
|---|---|---|
| Установка Python | Вручную, системный | Автоматически в папку проекта |
| Язык интерфейса | Только EN/ZH | **RU / EN** (переключатель) |
| Отладочное окно | Зависает при переключении | Исправлено |
| Картинка при забросе | Подвисает | Стабильный FPS |
| Захват экрана | Базовый | Оптимизирован (GDI reuse) |
| YOLO инференс | Стандартный | FP16 + настраиваемый imgsz |
| Китайские символы | Есть везде | Полностью убраны |

---

## Быстрый старт

### 1. Установка
Запустите `install.bat` и дождитесь завершения.

Скрипт автоматически:
- Скачает Python 3.10 в папку `python\` внутри проекта
- Определит наличие видеокарты NVIDIA и установит нужную версию PyTorch (CUDA или CPU)
- Установит все зависимости

Интернет нужен только при первой установке. После этого проект работает полностью автономно — системный Python не требуется.

### 2. Запуск
```
start.bat
```

---

## Требования

- Windows 10 / 11 (64-bit)
- Интернет при первом запуске install.bat (~500 МБ с CUDA PyTorch)
- VRChat с открытой игрой [fish!](https://vrchat.com/home/world/wrld_5d5a3e81-069d-4f5d-a962-e8c7bdcf4fbe)

> **GPU (NVIDIA)** ускоряет детекцию в ~3-5 раз, но не обязателен — работает и на CPU.
>
> Устанавливать CUDA Toolkit **не нужно** — PyTorch включает все CUDA-библиотеки в pip-пакете.
> Единственное требование для GPU-режима — **драйверы NVIDIA версии 570+** (обычно уже установлены).
> Проверить версию: `nvidia-smi` в командной строке.
>
> Если `install.bat` определил GPU неверно и поставил CPU-версию PyTorch — переустановить вручную:
> ```
> python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
> ```

---

## Использование

1. Запустите `start.bat`
2. Нажмите **«Найти окно»** — программа найдёт VRChat
3. (Опционально) Нажмите **«Выбрать зону»** и выделите область с мини-игрой
4. Нажмите **▶ Старт** или **F9**

### Горячие клавиши

| Клавиша | Действие |
|---|---|
| `F9` | Старт / Пауза |
| `F10` | Стоп |
| `F11` | Режим отладки |

---

## Структура проекта

```
vrchat-fish-portable/
├── python/           # Локальный Python (создаётся install.bat)
├── core/
│   ├── bot.py        # Основная логика бота
│   ├── screen.py     # Захват экрана (PrintWindow / mss)
│   ├── detector.py   # Детектор полоски и рыбы
│   ├── yolo_detector.py  # YOLO v8 детектор
│   ├── window.py     # Работа с окном VRChat
│   ├── input_ctrl.py # Управление вводом
│   └── overlay.py    # Отладочный оверлей
├── gui/
│   └── app.py        # Интерфейс (tkinter, RU/EN)
├── yolo/
│   └── best.pt       # Веса YOLO модели
├── utils/
│   └── logger.py     # Логгер
├── config.py         # Все настройки
├── main.py           # Точка входа
├── install.bat       # Установка
├── start.bat         # Запуск
└── requirements.txt
```

---

## Настройки

Все параметры доступны прямо в интерфейсе и применяются без перезапуска.

**Основные:**
| Параметр | Описание |
|---|---|
| Автоподсечка (с) | Через сколько секунд без поклёвки подсекать принудительно |
| Размер рыбы (px) | Размер иконки рыбы (влияет на масштаб поиска) |
| Мёртвая зона (px) | Зона нечувствительности контроллера |
| Мин./Макс. удержание (мс) | Диапазон времени зажатия кнопки |
| Пауза после вылова (с) | Ожидание перед следующим забросом |

**Расширенные (быстрый захват):**
Включает режим быстрой реакции когда рыба резко прыгает или движется на высокой скорости. Настраивается порог скорости, прыжка, усиление и т.д.

---

## Исправленные баги (по сравнению с оригиналом)

- **Зависание при переключении отладки** — `cv2.imshow` вызывался из разных потоков, что приводило к deadlock. Исправлено через `threading.Event` и единый поток для всех GUI-вызовов OpenCV.
- **Заморозка картинки при забросе/подъёме** — блокирующие `time.sleep()` прерывали обновление оверлея. Заменены на `_sleep_with_overlay()` который обновляет кадр в фоне.
- **Все китайские символы** — удалены из интерфейса, логов и исходного кода.

---

## Оптимизации

- **Захват экрана**: переиспользование GDI ресурсов (DC, bitmap, буфер) между кадрами вместо создания/удаления на каждый кадр
- **Проверка чёрного кадра**: замена `np.mean()` на 3-пиксельную выборку (~200× быстрее)
- **YOLO**: поддержка FP16 half-precision на GPU, настраиваемый `imgsz` (меньше = быстрее)
- **FPS счётчик**: `deque(maxlen=30)` вместо ручного среза списка

---

## Кредиты

- **Оригинальный автор**: [FoxieBoo](https://github.com/FoxieBoo) — [vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)
- **Этот форк**: портабельность, RU/EN интерфейс, исправления и оптимизации

---

---

# VRChat Fish — Portable (English)

**Portable fork** of the VRChat auto-fishing bot for the [fish!](https://vrchat.com/home/world/wrld_5d5a3e81-069d-4f5d-a962-e8c7bdcf4fbe) world.

> **Based on the original project by [FoxieBoo/vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)**
> This fork adds portable installation, bug fixes, and a Russian/English UI.

---

## What's different from the original

| | Original | This fork |
|---|---|---|
| Python installation | Manual, system-wide | Auto-downloaded into project folder |
| UI language | EN/ZH only | **RU / EN** (switchable) |
| Debug window | Freezes on toggle | Fixed |
| Overlay during cast | Stutters / freezes | Stable FPS |
| Screen capture | Basic | Optimized (GDI resource reuse) |
| YOLO inference | Default | FP16 + configurable imgsz |
| Chinese characters | Throughout | Fully removed |

---

## Quick start

### 1. Install
Run `install.bat` and wait for it to finish.

The script will automatically:
- Download Python 3.10 into the `python\` folder inside the project
- Detect NVIDIA GPU and install the correct PyTorch build (CUDA or CPU)
- Install all dependencies

Internet is only needed on first run. After that the project is fully self-contained — no system Python required.

### 2. Run
```
start.bat
```

---

## Requirements

- Windows 10 / 11 (64-bit)
- Internet on first `install.bat` run (~500 MB with CUDA PyTorch)
- VRChat with the [fish!](https://vrchat.com/home/world/wrld_5d5a3e81-069d-4f5d-a962-e8c7bdcf4fbe) world open

> **NVIDIA GPU** speeds up detection ~3-5x, but is not required — CPU mode works fine.
>
> You do **NOT** need to install CUDA Toolkit — PyTorch bundles all CUDA libraries in the pip package.
> The only requirement for GPU mode is **NVIDIA driver version 570+** (usually already installed).
> Check your version: run `nvidia-smi` in the command prompt.
>
> If `install.bat` installed the CPU version of PyTorch by mistake — reinstall manually:
> ```
> python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
> ```

---

## Usage

1. Run `start.bat`
2. Click **Connect window** — the program will find VRChat
3. (Optional) Click **Select ROI** and draw a rectangle around the mini-game area
4. Click **▶ Start** or press **F9**

### Hotkeys

| Key | Action |
|---|---|
| `F9` | Start / Pause |
| `F10` | Stop |
| `F11` | Debug mode |

---

## Bug fixes (vs original)

- **Debug window freeze on toggle** — `cv2.imshow` was called from multiple threads causing deadlock. Fixed with `threading.Event` and a dedicated thread for all OpenCV GUI calls.
- **Overlay freeze during cast/hook** — blocking `time.sleep()` calls stopped overlay updates. Replaced with `_sleep_with_overlay()` that keeps refreshing frames in the background.
- **Chinese characters** — removed from UI, logs, and all source files.

## Optimizations

- **Screen capture**: GDI resources (DC, bitmap, buffer) are pre-allocated and reused across frames instead of being created/destroyed per frame
- **Black frame check**: replaced `np.mean()` with 3-pixel sampling (~200× faster)
- **YOLO**: FP16 half-precision support on GPU, configurable `imgsz` (smaller = faster)
- **FPS counter**: `deque(maxlen=30)` instead of manual list slicing

---

## Credits

- **Original author**: [FoxieBoo](https://github.com/FoxieBoo) — [vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)
- **This fork**: portable setup, RU/EN interface, bug fixes and performance improvements