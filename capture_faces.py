import cv2
import os

# Ask for student details
student_id = input("Enter student ID (e.g. 101): ").strip()
student_name = input("Enter student name (e.g. Shivang): ").strip()

# Folder where images will be saved
folder_name = f"{student_id}_{student_name}"
save_path = os.path.join("faces", folder_name)

os.makedirs(save_path, exist_ok=True)

# Load Haar Cascade for face detection
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

print("[INFO] Press 'c' to capture face, 'q' to quit.")
img_count = 0
max_images = 50  # how many samples per student

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not found.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.putText(frame, f"Images: {img_count}/{max_images}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Capture Faces - Press 'c' to save face, 'q' to exit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save detected faces
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(save_path, f"{student_id}_{img_count}.jpg")
            cv2.imwrite(img_path, face_img)
            img_count += 1
            print(f"[INFO] Saved {img_path}")

            if img_count >= max_images:
                print("[INFO] Reached max images.")
                break

    elif key == ord('q'):
        break

    if img_count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done capturing faces.")
