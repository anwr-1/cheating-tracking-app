import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from tkinter import Tk, Button, Label, Frame, filedialog, messagebox, StringVar
from tkinter import ttk

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

MODEL_PATH = "best.pt"            # your trained cheating-detection YOLO weights
CHEATING_CLASS_NAME = "cheating"  # must match the class name your model was trained with
SCREENSHOT_DIR = "alerts/screenshots"
DB_PATH = "alerts/alerts.db"
CONF_THRESHOLD = 0.5
SCREENSHOT_COOLDOWN_SECONDS = 5   # avoid spamming a screenshot every frame for the same event


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            behavior TEXT NOT NULL,
            confidence REAL NOT NULL,
            screenshot_path TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_alert(behavior, confidence, screenshot_path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO alerts (timestamp, behavior, confidence, screenshot_path) VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), behavior, confidence, screenshot_path),
    )
    conn.commit()
    conn.close()


def fetch_alerts():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, behavior, confidence, screenshot_path FROM alerts ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


class CheatingTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cheating Tracking & Alert System")
        self.root.geometry("1000x650")

        self.model = None
        self.processing = False
        self.last_screenshot_time = 0

        self._build_ui()
        init_db()
        self._refresh_history()

    def _build_ui(self):
        self.top = Frame(self.root, pady=10)
        self.top.pack(fill="x")

        Button(self.top, text="📁 Upload Video", command=self.upload_video, width=18).pack(side="left", padx=10)
        self.status_var = StringVar(value="Ready. Upload an exam video to begin.")
        Label(self.top, textvariable=self.status_var, fg="#2c3e50").pack(side="left", padx=10)

        self.notification_var = StringVar(value="")
        self.notification_label = Label(
            self.root, textvariable=self.notification_var, fg="white", bg="#c0392b",
            font=("Segoe UI", 11, "bold"), pady=6
        )
        # not packed until a cheating alert fires

        main = Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        self.video_label = Label(main, bg="black")
        self.video_label.pack(side="left", fill="both", expand=True)

        history_frame = Frame(main, width=380)
        history_frame.pack(side="right", fill="y", padx=(10, 0))
        Label(history_frame, text="Alert History (double-click to open screenshot)",
              font=("Segoe UI", 10, "bold"), wraplength=350).pack(anchor="w")

        columns = ("time", "behavior", "confidence")
        self.tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=25)
        for col, label in zip(columns, ("Time", "Behavior", "Confidence")):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=110)
        self.tree.pack(fill="y", expand=True)
        self.tree.bind("<Double-1>", self._open_selected_screenshot)

    def upload_video(self):
        if self.processing:
            messagebox.showinfo("Busy", "A video is already being processed.")
            return

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Model not found",
                f"Could not find '{MODEL_PATH}'. Place your trained YOLO weights next to main.py."
            )
            return

        path = filedialog.askopenfilename(
            title="Select exam video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        if self.model is None:
            self.status_var.set("Loading model...")
            self.root.update_idletasks()
            self.model = YOLO(MODEL_PATH)

        self.processing = True
        threading.Thread(target=self._process_video, args=(path,), daemon=True).start()

    def _process_video(self, video_path):
        self.status_var.set(f"Processing {os.path.basename(video_path)}...")
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        # model.track() uses ByteTrack by default — assigns a persistent ID per student
        results_gen = self.model.track(
            source=video_path, stream=True, persist=True, conf=CONF_THRESHOLD, verbose=False
        )

        for result in results_gen:
            frame = result.orig_img.copy()
            cheating_detected = False
            best_conf = 0.0

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    is_cheating = cls_name.lower() == CHEATING_CLASS_NAME.lower()
                    color = (0, 0, 255) if is_cheating else (0, 200, 0)  # BGR: red / green

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.0%}"
                    cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if is_cheating:
                        cheating_detected = True
                        best_conf = max(best_conf, conf)

            if cheating_detected and (time.time() - self.last_screenshot_time) > SCREENSHOT_COOLDOWN_SECONDS:
                self.last_screenshot_time = time.time()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"cheating_{ts}.jpg")
                cv2.imwrite(screenshot_path, frame)
                log_alert("Cheating detected", best_conf, screenshot_path)
                self.root.after(0, self._on_new_alert, best_conf)

            self._show_frame(frame)
            time.sleep(0.01)  # keep UI responsive

        self.processing = False
        self.root.after(0, lambda: self.status_var.set("Done. Ready for another video."))

    def _show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((700, 500))
        imgtk = ImageTk.PhotoImage(image=img)

        def update():
            self.video_label.imgtk = imgtk  # keep a reference so it isn't garbage collected
            self.video_label.configure(image=imgtk)

        self.root.after(0, update)

    def _on_new_alert(self, confidence):
        self.notification_var.set(
            f"🚨 Cheating Alert Detected — Behavior: Cheating — "
            f"Time: {datetime.now().strftime('%I:%M %p')} — Confidence: {confidence:.0%}"
        )
        self.notification_label.pack(fill="x", after=self.top)
        self.root.after(4000, self.notification_label.pack_forget)
        self._refresh_history()

    def _refresh_history(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        if not os.path.exists(DB_PATH):
            return
        for ts, behavior, conf, path in fetch_alerts():
            self.tree.insert("", "end", values=(ts, behavior, f"{conf:.0%}"))

    def _open_selected_screenshot(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        ts = self.tree.item(selected[0])["values"][0]
        for row_ts, behavior, conf, path in fetch_alerts():
            if row_ts == ts and os.path.exists(path):
                open_file(path)
                break


if __name__ == "__main__":
    root = Tk()
    app = CheatingTrackerApp(root)
    root.mainloop()
