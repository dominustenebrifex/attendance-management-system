import cv2
import os
import pickle
from datetime import datetime
import pandas as pd

cascade_path = "haarcascade_frontalface_default.xml"
model_path = os.path.join("models", "face_model.yml")
labels_path = os.path.join("models", "labels.pickle")

attendance_dir = "attendance"
os.makedirs(attendance_dir, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load labels
with open(labels_path, "rb") as f:
    label_map = pickle.load(f)  # id -> "101_Name"

# For storing who is already marked present in this session
marked_present = set()

# Create today's CSV file
today_str = datetime.now().strftime("%Y-%m-%d")
csv_path = os.path.join(attendance_dir, f"attendance_{today_str}.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["StudentID", "Name", "Time"])

cap = cv2.VideoCapture(0)

print("[INFO] Press 'q' to quit attendance system.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(face_roi)

        # Lower confidence = better match, adjust threshold if needed
        if confidence < 70:  # try 60–80
            person_folder = label_map[label_id]  # "101_Shivang"
            student_id, student_name = person_folder.split("_", 1)

            text = f"{student_name} ({int(confidence)})"
            color = (255, 255, 255)

            # Mark attendance if not already
            if student_id not in marked_present:
                time_str = datetime.now().strftime("%H:%M:%S")
                df.loc[len(df)] = [student_id, student_name, time_str]
                df.to_csv(csv_path, index=False)
                marked_present.add(student_id)
                print(f"[ATTENDANCE] Marked present: {student_name} at {time_str}")
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imshow("Face Recognition Attendance - Press 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance system closed.")
print(f"[INFO] Attendance saved to: {csv_path}")
