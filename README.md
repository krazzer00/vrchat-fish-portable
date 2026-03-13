<div align="center">
# 🎣 VRChat Fish — Portable
**Портабельный авто-бот для рыбалки в VRChat (мир fish!)**
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)
[![YOLO](https://img.shields.io/badge/YOLOv8-Detection-FF4B4B?style=for-the-badge&logo=opencv&logoColor=white)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Based on](https://img.shields.io/badge/Based%20on-FoxieBoo%2Fvrchat--fish-purple?style=for-the-badge&logo=github)](https://github.com/FoxieBoo/vrchat-fish)
[🇷🇺 Русский](#-быстрый-старт) · [🇬🇧 English](#-quick-start-english)
</div>
---
## 📖 О проекте
Портабельная версия автоматической рыбалки для VRChat (игра **fish!**).  
Форк оригинального проекта [FoxieBoo/vrchat-fish](https://github.com/FoxieBoo/vrchat-fish) с портабельной установкой, исправлениями багов и интерфейсом на русском и английском языках.
---
## ⚡ Отличия от оригинала
| Функция | Оригинал | Этот форк |
|---|---|---|
| 🐍 Установка Python | Вручную, системный | ✅ Автоматически в папку проекта |
| 🌐 Язык интерфейса | Только EN/ZH | ✅ RU / EN (переключатель) |
| 🪟 Отладочное окно | Зависает при переключении | ✅ Исправлено |
| 🖼️ Картинка при забросе | Подвисает | ✅ Стабильный FPS |
| 📸 Захват экрана | Базовый | ✅ Оптимизирован (GDI reuse) |
| 🤖 YOLO инференс | Стандартный | ✅ FP16 + настраиваемый imgsz |
| 🀄 Китайские символы | Есть везде | ✅ Полностью убраны |
---
## 🚀 Быстрый старт
### 1. Установка
Запустите `install.bat` и дождитесь завершения.
Скрипт автоматически:
- 📥 Скачает **Python 3.10** в папку `python\\` внутри проекта
- 🖥️ Определит наличие видеокарты NVIDIA и установит нужную версию **PyTorch** (CUDA или CPU)
- 📦 Установит все зависимости
> 💡 Интернет нужен только при первой установке. После этого проект работает полностью автономно — системный Python не требуется.
### 2. Запуск
```bat
start.bat
```
---
## 💻 Системные требования
| Требование | Описание |
|---|---|
| 🪟 ОС | Windows 10 / 11 (64-bit) |
| 🌐 Интернет | Только при первом запуске install.bat (~500 МБ с CUDA PyTorch) |
| 🎮 VRChat | С открытой игрой fish! |
| 🎮 GPU (опционально) | NVIDIA ускоряет детекцию в **~3–5 раз** |
> **GPU-режим:** Не нужно устанавливать CUDA Toolkit — PyTorch включает все CUDA-библиотеки.  
> Единственное требование — **драйверы NVIDIA версии 570+**.  
> Проверить: `nvidia-smi` в командной строке.
<details>
<summary>⚠️ Если install.bat неверно определил GPU</summary>
```bat
python\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
</details>
---
## 🕹️ Использование
1. Запустите `start.bat`
2. Нажмите **«Найти окно»** — программа найдёт VRChat
3. *(Опционально)* Нажмите **«Выбрать зону»** и выделите область с мини-игрой
4. Нажмите **▶ Старт** или `F9`
### ⌨️ Горячие клавиши
| Клавиша | Действие |
|---|---|
| `F9` | ▶ Старт / ⏸ Пауза |
| `F10` | ⏹ Стоп |
| `F11` | 🐛 Режим отладки |
---
## ⚙️ Настройки
Все параметры доступны прямо в интерфейсе и применяются **без перезапуска**.
| Параметр | Описание |
|---|---|
| Автоподсечка (с) | Через сколько секунд без поклёвки подсекать принудительно |
| Размер рыбы (px) | Размер иконки рыбы (влияет на масштаб поиска) |
| Мёртвая зона (px) | Зона нечувствительности контроллера |
| Мин./Макс. удержание (мс) | Диапазон времени зажатия кнопки |
| Пауза после вылова (с) | Ожидание перед следующим забросом |
<details>
<summary>🔬 Расширенные настройки (быстрый захват)</summary>
Включает режим быстрой реакции, когда рыба резко прыгает или движется на высокой скорости.  
Настраивается порог скорости, прыжка, усиление и т.д.
</details>
---
## 🗂️ Структура проекта
```
vrchat-fish-portable/
├── 🐍 python/              # Локальный Python (создаётся install.bat)
├── 📁 core/
│   ├── bot.py              # Основная логика бота
│   ├── screen.py           # Захват экрана (PrintWindow / mss)
│   ├── detector.py         # Детектор полоски и рыбы
│   ├── yolo_detector.py    # YOLO v8 детектор
│   ├── window.py           # Работа с окном VRChat
│   ├── input_ctrl.py       # Управление вводом
│   └── overlay.py          # Отладочный оверлей
├── 📁 gui/
│   └── app.py              # Интерфейс (tkinter, RU/EN)
├── 📁 yolo/
│   └── best.pt             # Веса YOLO модели
├── 📁 utils/
│   └── logger.py           # Логгер
├── config.py               # Все настройки
├── main.py                 # Точка входа
├── install.bat             # 🔧 Установка
├── start.bat               # 🚀 Запуск
└── requirements.txt
```
---
## 🐛 Исправленные баги
<details>
<summary>Посмотреть список</summary>
- **Зависание при переключении отладки** — `cv2.imshow` вызывался из разных потоков, что приводило к deadlock. Исправлено через `threading.Event` и единый поток для всех GUI-вызовов OpenCV.
- **Заморозка картинки при забросе/подъёме** — блокирующие `time.sleep()` прерывали обновление оверлея. Заменены на `_sleep_with_overlay()`, который обновляет кадр в фоне.
- **Китайские символы** — полностью удалены из интерфейса, логов и исходного кода.
</details>
---
## 🔧 Оптимизации
- **📸 Захват экрана:** переиспользование GDI ресурсов (DC, bitmap, буфер) между кадрами вместо создания/удаления на каждый кадр
- **⬛ Проверка чёрного кадра:** замена `np.mean()` на 3-пиксельную выборку (~**200× быстрее**)
- **🤖 YOLO:** поддержка FP16 half-precision на GPU, настраиваемый `imgsz` (меньше = быстрее)
- **📊 FPS счётчик:** `deque(maxlen=30)` вместо ручного среза списка
---
## 👏 Кредиты
- **Оригинальный автор:** [FoxieBoo](https://github.com/FoxieBoo) — [vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)
- **Этот форк:** портабельность, RU/EN интерфейс, исправления и оптимизации
---
<div align="center">
---
# 🎣 VRChat Fish — Portable (English)
**Portable auto-fishing bot for VRChat (fish! world)**
</div>
## 📖 About
Portable fork of the VRChat auto-fishing bot for the **fish!** world.  
Based on the original project by [FoxieBoo/vrchat-fish](https://github.com/FoxieBoo/vrchat-fish).  
This fork adds portable installation, bug fixes, and a Russian/English UI.
---
## ⚡ What's Different
| Feature | Original | This Fork |
|---|---|---|
| 🐍 Python installation | Manual, system-wide | ✅ Auto-downloaded into project folder |
| 🌐 UI language | EN/ZH only | ✅ RU / EN (switchable) |
| 🪟 Debug window | Freezes on toggle | ✅ Fixed |
| 🖼️ Overlay during cast | Stutters / freezes | ✅ Stable FPS |
| 📸 Screen capture | Basic | ✅ Optimized (GDI resource reuse) |
| 🤖 YOLO inference | Default | ✅ FP16 + configurable imgsz |
| 🀄 Chinese characters | Throughout | ✅ Fully removed |
---
## 🚀 Quick Start (English)
### 1. Install
Run `install.bat` and wait for it to finish. The script will automatically:
- 📥 Download **Python 3.10** into the `python\\` folder inside the project
- 🖥️ Detect NVIDIA GPU and install the correct **PyTorch** build (CUDA or CPU)
- 📦 Install all dependencies
> 💡 Internet is only needed on first run. After that the project is fully self-contained — no system Python required.
### 2. Run
```bat
start.bat
```
---
## 💻 Requirements
| Requirement | Details |
|---|---|
| 🪟 OS | Windows 10 / 11 (64-bit) |
| 🌐 Internet | First install only (~500 MB with CUDA PyTorch) |
| 🎮 VRChat | With the fish! world open |
| 🎮 GPU (optional) | NVIDIA speeds up detection **~3–5x** |
> **GPU mode:** You do NOT need to install CUDA Toolkit — PyTorch bundles all CUDA libraries.  
> The only requirement is **NVIDIA driver version 570+**.  
> Check: run `nvidia-smi` in the command prompt.
<details>
<summary>⚠️ If install.bat installed the wrong PyTorch build</summary>
```bat
python\\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
</details>
---
## 🕹️ Usage
1. Run `start.bat`
2. Click **Connect window** — the program will find VRChat
3. *(Optional)* Click **Select ROI** and draw a rectangle around the mini-game area
4. Click **▶ Start** or press `F9`
### ⌨️ Hotkeys
| Key | Action |
|---|---|
| `F9` | ▶ Start / ⏸ Pause |
| `F10` | ⏹ Stop |
| `F11` | 🐛 Debug mode |
---
## 🐛 Bug Fixes
<details>
<summary>View list</summary>
- **Debug window freeze on toggle** — `cv2.imshow` was called from multiple threads causing deadlock. Fixed with `threading.Event` and a dedicated thread for all OpenCV GUI calls.
- **Overlay freeze during cast/hook** — blocking `time.sleep()` calls stopped overlay updates. Replaced with `_sleep_with_overlay()` that keeps refreshing frames in the background.
- **Chinese characters** — removed from UI, logs, and all source files.
</details>
---
## 🔧 Optimizations
- **📸 Screen capture:** GDI resources (DC, bitmap, buffer) are pre-allocated and reused across frames instead of being created/destroyed per frame
- **⬛ Black frame check:** replaced `np.mean()` with 3-pixel sampling (~**200× faster**)
- **🤖 YOLO:** FP16 half-precision support on GPU, configurable `imgsz` (smaller = faster)
- **📊 FPS counter:** `deque(maxlen=30)` instead of manual list slicing
---
## 👏 Credits
- **Original author:** [FoxieBoo](https://github.com/FoxieBoo) — [vrchat-fish](https://github.com/FoxieBoo/vrchat-fish)
- **This fork:** portable setup, RU/EN interface, bug fixes and performance improvements
---
<div align="center">
Made with ❤️ for the VRChat fishing community
</div>
