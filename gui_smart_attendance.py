import cv2
import os
import numpy as np
import pickle
import csv
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

# ================= CONFIG =================

APP_NAME = "NEEV SmartAttend"
SCHOOL_NAME = "Delhi Govt. School"

# Colors for UI
BG = "#0f172a"
CARD = "#1f2933"
ACCENT = "#38bdf8"
TEXT = "#e5e7eb"

# Paths
CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACES_DIR = "faces"
MODELS_DIR = "models"
ATT_DIR = "attendance"
MODEL_PATH = os.path.join(MODELS_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.pickle")

# Make sure folders exist
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ATT_DIR, exist_ok=True)

# ================= CAMERA HELPER =================

def open_camera():
    """Try different camera indexes so it works on most systems."""
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None

# ================= REGISTER / CAPTURE FACES =================

def capture_faces(student_id, student_name, status_var):
    folder_name = f"{student_id}_{student_name}"
    save_path = os.path.join(FACES_DIR, folder_name)
    os.makedirs(save_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = open_camera()

    if cap is None:
        messagebox.showerror("Error", "Camera not found. Close other apps using camera.")
        status_var.set("❌ Camera not found.")
        return

    messagebox.showinfo(
        "Instructions",
        "Camera khulne ke baad:\n"
        "- Window pe click karo\n"
        "- 'c' dabao face capture ke liye\n"
        "- 'q' dabao jab kaam ho jaye"
    )

    status_var.set(f"📸 Capturing faces for {student_name}...")
    img_count = 0
    max_images = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.putText(
            frame,
            f"{img_count}/{max_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Capture Faces (c = capture, q = quit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                img_path = os.path.join(save_path, f"{student_id}_{img_count}.jpg")
                cv2.imwrite(img_path, face_img)
                img_count += 1
                if img_count >= max_images:
                    break

        if key == ord("q") or img_count >= max_images:
            break

    cap.release()
    cv2.destroyAllWindows()

    status_var.set(f"✅ Captured {img_count} images for {student_name}.")
    messagebox.showinfo("Done", f"Captured {img_count} images for {student_name}.")

def register_student_gui(status_var):
    sid = simpledialog.askstring("Student ID", "Enter Student ID (e.g. 101):")
    if not sid:
        return
    name = simpledialog.askstring("Student Name", "Enter Student Name (e.g. Shivang):")
    if not name:
        return

    sid = sid.strip()
    name = name.strip()

    if not sid or not name:
        messagebox.showerror("Error", "ID and Name cannot be empty.")
        return

    capture_faces(sid, name, status_var)

# ================= TRAIN MODEL =================

def train_model(status_var):
    face_images = []
    ids = []
    label_map = {}
    label_id = 0

    # Scan all folders inside faces/
    for person_folder in os.listdir(FACES_DIR):
        person_path = os.path.join(FACES_DIR, person_folder)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_folder

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            face_images.append(img)
            ids.append(label_id)

        label_id += 1

    if len(face_images) == 0:
        messagebox.showerror("Error", "No faces found. Pehle students register karo.")
        status_var.set("⚠️ No faces found.")
        return

    status_var.set("🧠 Training model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_images, np.array(ids))
    recognizer.write(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    status_var.set("✅ Model trained successfully.")
    messagebox.showinfo("Success", "Model trained and saved successfully.")

# ================= START ATTENDANCE =================

def start_attendance(status_var):
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        messagebox.showerror("Error", "Model not found. Pehle Train Model dabao.")
        status_var.set("⚠️ Model missing.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = open_camera()

    if cap is None:
        messagebox.showerror("Error", "Camera not found.")
        status_var.set("❌ Camera not found.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATT_DIR, f"attendance_{today}.csv")

    # Make CSV if not exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["StudentID", "Name", "Time"])

    marked = set()
    messagebox.showinfo(
        "Instructions",
        "Attendance started:\n"
        "- Window pe click karo\n"
        "- Students camera ke saamne aayein\n"
        "- 'q' dabao jab kaam ho jaye"
    )

    status_var.set("✅ Attendance started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            try:
                label_id, confidence = recognizer.predict(face_roi)
            except:
                continue

            if confidence < 70:
                folder_name = label_map[label_id]      # "101_Shivang"
                student_id, student_name = folder_name.split("_", 1)
                display_name = student_name
                color = (255, 255, 255)

                if student_id not in marked:
                    marked.add(student_id)
                    time_now = datetime.now().strftime("%H:%M:%S")
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([student_id, student_name, time_now])
                    status_var.set(f"📌 Marked present: {student_name} at {time_now}")
            else:
                display_name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                display_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Attendance (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    status_var.set(f"📂 Attendance saved: {csv_path}")
    messagebox.showinfo("Done", f"Attendance saved to:\n{csv_path}")

# ================= VIEW TODAY'S ATTENDANCE =================

def view_today_attendance(status_var):
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(ATT_DIR, f"attendance_{today}.csv")

    if not os.path.exists(csv_path):
        messagebox.showinfo("Info", "Aaj ke liye koi attendance nahi mili.")
        status_var.set("ℹ️ No attendance for today.")
        return

    win = tk.Toplevel()
    win.title(f"Today's Attendance - {today}")
    win.geometry("420x300")

    tree = ttk.Treeview(win, columns=("id", "name", "time"), show="headings")
    tree.heading("id", text="Student ID")
    tree.heading("name", text="Name")
    tree.heading("time", text="Time")

    tree.column("id", width=80, anchor="center")
    tree.column("name", width=200, anchor="w")
    tree.column("time", width=100, anchor="center")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            tree.insert("", "end", values=row)

    tree.pack(fill="both", expand=True)

# ================= GUI SETUP =================

def main():
    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("520x460")
    root.configure(bg=BG)
    root.resizable(False, False)

    status_var = tk.StringVar(value="Ready.")

    # Title
    tk.Label(
        root,
        text=APP_NAME,
        font=("Segoe UI", 18, "bold"),
        fg=ACCENT,
        bg=BG
    ).pack(pady=(10, 2))

    tk.Label(
        root,
        text=f"{SCHOOL_NAME} • Face Attendance",
        font=("Segoe UI", 10),
        fg=TEXT,
        bg=BG
    ).pack(pady=(0, 10))

    # Card frame
    card = tk.Frame(root, bg=CARD, bd=0, relief="ridge")
    card.pack(padx=20, pady=10, fill="both", expand=True)

    btn_frame = tk.Frame(card, bg=CARD)
    btn_frame.pack(pady=20)

    btn_style = {
        "font": ("Segoe UI", 11, "bold"),
        "bg": ACCENT,
        "fg": "#000000",
        "activebackground": "#0ea5e9",
        "activeforeground": "#000000",
        "bd": 0,
        "relief": "flat",
        "width": 26,
        "height": 2,
        "cursor": "hand2"
    }

    tk.Button(
        btn_frame,
        text="🧑‍🎓  Register New Student",
        command=lambda: register_student_gui(status_var),
        **btn_style
    ).pack(pady=5)

    tk.Button(
        btn_frame,
        text="🧠  Train Model",
        command=lambda: train_model(status_var),
        **btn_style
    ).pack(pady=5)

    tk.Button(
        btn_frame,
        text="✅  Start Attendance",
        command=lambda: start_attendance(status_var),
        **btn_style
    ).pack(pady=5)

    tk.Button(
        btn_frame,
        text="📊  View Today's Attendance",
        command=lambda: view_today_attendance(status_var),
        **btn_style
    ).pack(pady=5)

    # Status label
    tk.Label(
        root,
        textvariable=status_var,
        font=("Segoe UI", 9),
        fg=TEXT,
        bg=BG
    ).pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    main()
