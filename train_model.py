import cv2
import os
import numpy as np
import pickle

faces_dir = "faces"
model_path = os.path.join("models", "face_model.yml")
labels_path = os.path.join("models", "labels.pickle")

os.makedirs("models", exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_samples = []
ids = []

label_id = 0
label_map = {}  # numeric_id -> "101_Shivang"

for person_folder in os.listdir(faces_dir):
    person_path = os.path.join(faces_dir, person_folder)
    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person_folder  # store folder name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        face_samples.append(img)
        ids.append(label_id)

    label_id += 1

if len(face_samples) == 0:
    print("[ERROR] No face images found. Capture faces first.")
    exit()

print("[INFO] Training model... This may take a few seconds.")
recognizer.train(face_samples, np.array(ids))

recognizer.write(model_path)
print(f"[INFO] Model saved to {model_path}")

with open(labels_path, "wb") as f:
    pickle.dump(label_map, f)

print(f"[INFO] Labels saved to {labels_path}")
