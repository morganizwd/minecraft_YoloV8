import cv2, numpy as np, mss, pygetwindow as gw, tkinter as tk, keyboard, threading
import win32api, win32con, torch, math, time
from tkinter import messagebox
from ultralytics import YOLO

# ── ПУТИ К ВЕСАМ ────────────────────────────────────────────────────────────────
PRIMARY_MODEL_PATH   = r"C:/Users/morga/Desktop/minecraft_v2/runs/train_v22/weights/best.pt"
SECONDARY_MODEL_PATH = r"C:/Users/morga/Desktop/minecraft_v2/runs/train_v2/weights/best.pt"

# ── НАСТРОЙКИ ПО УМОЛЧАНИЮ (меняются слайдерами) ───────────────────────────────
DEFAULT_FOV_RADIUS  = 120      # px
DEFAULT_SENSITIVITY = 1.0      # множитель dx/dy
DEFAULT_COOLDOWN_MS = 80       # мс между движениями

IMG_SIZE   = 640
CONF_THRES = 0.25
IOU_NMS    = 0.5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── ГЛОБАЛЬНОЕ СОСТОЯНИЕ ────────────────────────────────────────────────────────
enable_aim   = True
last_move_ms = 0

# ── МОДЕЛИ ──────────────────────────────────────────────────────────────────────
model1 = YOLO(PRIMARY_MODEL_PATH)
model2 = YOLO(SECONDARY_MODEL_PATH)

# ── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────────────────────────────────
def capture_window(title):
    try:
        w = gw.getWindowsWithTitle(title)[0]
        if w.isMinimized or w.width == 0 or w.height == 0:
            raise IndexError
        mon = {"top": w.top, "left": w.left, "width": w.width, "height": w.height}
        with mss.mss() as sct:
            raw = sct.grab(mon)
            img = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        return img, mon
    except IndexError:
        messagebox.showerror("Ошибка", f"Окно «{title}» не найдено/свернуто")
        return None, None

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / float(areaA + areaB - inter + 1e-6)

def merge_boxes(b, c):
    keep, idxs = [], np.argsort(-c)
    while len(idxs):
        i = idxs[0]; keep.append(i); idxs = idxs[1:]
        idxs = [j for j in idxs if iou(b[i], b[j]) < IOU_NMS]
    return b[keep], c[keep]

# ── ГОРЯЧАЯ КЛАВИША ────────────────────────────────────────────────────────────
def toggle_aim():
    global enable_aim
    enable_aim = not enable_aim
    status_var.set(f"Авто‑прицел: {'ON' if enable_aim else 'OFF'}  |  FOV {fov_var.get()}  |  S {sens_var.get():.2f}  |  cd {cool_var.get():.0f} мс")
keyboard.add_hotkey("ctrl+alt+a", toggle_aim)

# ── ОСНОВНОЙ ЦИКЛ ДЕТЕКЦИИ ─────────────────────────────────────────────────────
def detection_loop(title):
    global last_move_ms
    while app.is_running:
        frame, mon = capture_window(title)
        if frame is None: break

        h, w = frame.shape[:2]
        cx_win, cy_win = w // 2, h // 2
        fov_r   = fov_var.get()
        sens    = sens_var.get()
        cd_ms   = cool_var.get()

        cv2.circle(frame, (cx_win, cy_win), int(fov_r), (255,255,0), 1)

        r1 = model1.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE, verbose=False)[0]
        r2 = model2.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, device=DEVICE, verbose=False)[0]

        boxes, confs = [], []
        for r in (r1, r2):
            if r.boxes and len(r.boxes):
                boxes.append(r.boxes.xyxy.cpu().numpy())
                confs.append(r.boxes.conf.cpu().numpy())
        if boxes:
            boxes = np.vstack(boxes).astype(int)
            confs = np.concatenate(confs)
            boxes, confs = merge_boxes(boxes, confs)
        else:
            boxes, confs = np.empty((0,4),int), np.array([])

        target = None
        for (x1,y1,x2,y2), cf in zip(boxes, confs):
            xc, yc = (x1+x2)//2, (y1+y2)//2
            dist = math.hypot(xc-cx_win, yc-cy_win)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0) if dist<=fov_r else (0,0,255),1)
            if dist<=fov_r and (target is None or cf>target[2]):
                target = (xc, yc, cf)

        if target and enable_aim:
            now = time.perf_counter()*1000
            if now - last_move_ms >= cd_ms:
                sx = mon["left"] + target[0]
                sy = mon["top"]  + target[1]
                cur_x, cur_y = win32api.GetCursorPos()
                dx = int((sx - cur_x) * sens)
                dy = int((sy - cur_y) * sens)
                if dx or dy:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
                    last_move_ms = now

        cv2.imshow("Minecraft Detection", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            app.is_running=False; break

    cv2.destroyAllWindows()
    app.is_running=False
    start_btn.config(text="Запустить")

# ── GUI ────────────────────────────────────────────────────────────────────────
app = tk.Tk(); app.title("Minecraft FOV‑Aim"); app.geometry("480x460"); app.is_running=False

# список окон
tk.Label(app,text="Окно для захвата:",font=("Arial",11)).pack(pady=4)
window_lb = tk.Listbox(app,width=60,height=8); window_lb.pack()
def refresh_windows():
    window_lb.delete(0,tk.END)
    for t in gw.getAllTitles():
        if t.strip(): window_lb.insert(tk.END,t)
tk.Button(app,text="Обновить список",command=refresh_windows).pack(pady=2)

# ── СЛАЙДЕРЫ ───────────────────────────────────────────────────────────────────
def make_slider(text,var,from_,to,step):
    frm = tk.Frame(app); frm.pack()
    tk.Label(frm,text=text,width=23,anchor='w').pack(side='left')
    tk.Scale(frm,variable=var,from_=from_,to=to,resolution=step,
             orient='horizontal',length=220).pack(side='left')

fov_var  = tk.IntVar(value=DEFAULT_FOV_RADIUS)
sens_var = tk.DoubleVar(value=DEFAULT_SENSITIVITY)
cool_var = tk.DoubleVar(value=DEFAULT_COOLDOWN_MS)

make_slider("Радиус FOV (px)",       fov_var,  40, 300,   5)
make_slider("Чувствительность",      sens_var, 0.3, 3.0,  0.05)
make_slider("Кулдаун (мс)",          cool_var, 10, 300,  10)

# статус
status_var = tk.StringVar()
status_var.set(f"Авто‑прицел: ON  |  FOV {fov_var.get()}  |  S {sens_var.get():.2f}  |  cd {cool_var.get():.0f} мс")
tk.Label(app,textvariable=status_var,fg="gray").pack(pady=4)

# запуск/стоп
def toggle_detection():
    if not app.is_running:
        sel = window_lb.curselection()
        if not sel:
            messagebox.showwarning("Окно","Сначала выберите окно!"); return
        title = window_lb.get(sel)
        app.is_running=True; start_btn.config(text="Остановить")
        threading.Thread(target=detection_loop,args=(title,),daemon=True).start()
    else:
        app.is_running=False; start_btn.config(text="Запустить")
start_btn = tk.Button(app,text="Запустить",command=toggle_detection); start_btn.pack(pady=10)

# обновлять строку статуса при изменении слайдеров
def update_status(*_):
    status_var.set(f"Авто‑прицел: {'ON' if enable_aim else 'OFF'}  |  "
                   f"FOV {fov_var.get()}  |  S {sens_var.get():.2f}  |  cd {cool_var.get():.0f} мс")
for v in (fov_var,sens_var,cool_var): v.trace_add("write",update_status)

refresh_windows()
app.mainloop()
