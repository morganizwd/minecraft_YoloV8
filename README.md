# 📦 Minecraft v2 — YOLOv8 + FOV‑Aimbot

**“Настрой и запусти за вечер (даже если ты никогда не видел Python)”**

---

## 0. О чём репозиторий

* **`realtime_minecraft_detection.py`** — GUI‑скрипт, который

  1. захватывает выбранное окно игры,
  2. пропускает картинку через **две** обученные YOLOv8‑модели,
  3. рисует боксы класса `player`,
  4. «магнитит» курсор к цели внутри жёлтого круга FOV.

* **`runs/train_v22/weights/best.pt`**  и **`runs/train_v2/weights/best.pt`** — веса двух сетей (основной и дополнительной).

* Файлы данных (`images/`, `labels/`, `.yaml`) — если захотите переобучить сеть.

---

## 1. Что понадобится

| Что               | Минимальные версии                                              |
| ----------------- | --------------------------------------------------------------- |
| **Windows 10/11** | (Linux можно, но нельзя)                                        |
| **Python**        | 3.9 – 3.11 (СТРОГО!)                                            |
| **NVIDIA GPU**    | любая RTX/GTX (желательно ≥ 8 GB VRAM)                          |
| **CUDA Toolkit**  | не обязателен, но ускоряет; поддержка SM 5.0+                   |
| **Git**           | чтобы клонировать репозиторий                                   |

> 💡 На CPU **тоже** запустится, но FPS будет \~ 0.5. На RTX 3060 Ti — 20-30 fps.

---

## 2. Клонируем репозиторий

```powershell
git clone https://github.com/morganizwd/minecraft_YoloV8
cd minecraft_YoloV8
```

## 3. Создаём изолированную среду

```powershell
python -m venv venv
.\\venv\\Scripts\\activate        
python -m pip install -U pip
```

## 4. Ставим зависимости

### 4.1 PyTorch + CUDA 11.8

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

(Если у тебя нет CUDA‑видеокарты — замени cu118 на cpu)
```

### 4.2 Остальное

````powershell
pip install ultralytics==8.3.* opencv-python==4.11.0.80 `
           mss pygetwindow keyboard pywin32 numpy

❗ Важно: Установился opencv-python, а не opencv-python-headless.
Если всё‑таки прилетел headless, удали и поставь обычный:
```powershell
pip uninstall opencv-python-headless -y && pip install opencv-python
````

## 5. Проверяем установку

```bash
python - << "PY"
import torch, cv2, ultralytics, win32api
print("CUDA:", torch.cuda.is_available())
print("OpenCV GUI:", "WIN32UI" in cv2.getBuildInformation())
print("Ultralytics ver:", ultralytics.__version__)
PY
```

Увидел CUDA: True, WIN32UI, версию Ultralytics — значит всё ок.

## 6. Запуск скрипта

```bash
python realtime_minecraft_detection.py
```

### 6.1 Что появится

* Окно Minecraft FOV‑Aim.
* Слева список всех текущих окон Windows.

### 6.2 Пошагово

| Шаг | Действие                                                                          |
| --- | --------------------------------------------------------------------------------- |
| 1   | Нажми «Обновить список».                                                          |
| 2   | Выбери стрелкой окно с игрой Minecraft (или любое, что хочешь трекать).           |
| 3   | Настрой слайдеры:                                                                 |
|     | • Радиус FOV — диаметр жёлтого круга.                                             |
|     | • Чувствительность — насколько резко движется курсор (1 = в точку, 0.5 = плавно). |
|     | • Кулдаун — пауза между рывками.                                                  |
| 4   | Жми «Запустить». Откроется окно с видео + боксами.                                |
| 5   | В любой момент Ctrl + Alt + A — вкл./выкл. притягивание мыши.                     |
| 6   | Закрыть можно клавишей q в окне видео или кнопкой «Остановить».                   |

## 7. Кастомизация моделей

Файлы весов лежат по умолчанию в
`runs/train_v22/weights/best.pt` (основная) и
`runs/train_v2/weights/best.pt` (дополнительная).

Хочешь другие — замени пути в начале `realtime_minecraft_detection.py`:

```python
PRIMARY_MODEL_PATH   = r"path\to\your\first_model.pt"
SECONDARY_MODEL_PATH = r"path\to\your\second_model.pt"
```

Класс всегда один (player). Если добавишь новые — придётся править скрипт.
