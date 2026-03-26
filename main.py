import cv2
import os
import time


class FaceDetector:
    def __init__(self):
        self.model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        if not os.path.exists(self.model_path):
            raise Exception("Haarcascade file not found!")

        self.face_cascade = cv2.CascadeClassifier(self.model_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,       # Better accuracy
            minNeighbors=6,        # Reduce false positives
            minSize=(40, 40)
        )
        return faces


def draw_faces(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def webcam_mode(detector):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    print("Press 'q' to quit")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        draw_faces(frame, faces)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Display info
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Face Detection (Webcam)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_mode(detector, path):
    if not os.path.exists(path):
        print("❌ Image not found")
        return

    image = cv2.imread(path)
    faces = detector.detect_faces(image)

    draw_faces(image, faces)

    print(f"✅ Faces detected: {len(faces)}")

    cv2.imshow("Face Detection (Image)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_mode(detector, path):
    if not os.path.exists(path):
        print("❌ Video not found")
        return

    cap = cv2.VideoCapture(path)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        draw_faces(frame, faces)

        cv2.imshow("Face Detection (Video)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    detector = FaceDetector()

    while True:
        print("\n===== FACE DETECTION SYSTEM =====")
        print("1. Webcam Detection")
        print("2. Image Detection")
        print("3. Video Detection")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            webcam_mode(detector)

        elif choice == "2":
            path = input("Enter image path: ")
            image_mode(detector, path)

        elif choice == "3":
            path = input("Enter video path: ")
            video_mode(detector, path)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
