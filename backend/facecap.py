import cv2
from ultralytics import YOLO
import requests
import hashlib
import logging

# Configuration
API_ENDPOINT = "http://localhost:8000/match_face/"
MODEL_PATH = "yolov8n-face-lindevs.pt"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s') 

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit(1)

# Access webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    logging.error("Failed to access the webcam.")
    exit(1)

saved_hashes = set()
logging.info("Face capture started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Unable to read from webcam.")
            break

        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > CONFIDENCE_THRESHOLD:
                cropped = frame[y1:y2, x1:x2]
                face_hash = hashlib.md5(cropped.tobytes()).hexdigest()
                
                if face_hash not in saved_hashes:
                    _, img_encoded = cv2.imencode('.jpg', cropped)

                    try:
                        response = requests.post(API_ENDPOINT, files={'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')})
                        data = response.json()
                        if response.status_code == 200 and data.get("status") == "matched":
                            name = data.get('name')
                            medical_id = data.get('medical_id')
                            logging.info(f"Matched: {name} | Medical ID: {medical_id}")
                        else:
                            logging.info("No face match found.")
                    except Exception as e:
                        logging.error(f"Error sending request: {e}")

                    saved_hashes.add(face_hash)

        cv2.imshow("YOLO Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logging.info("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Camera released. Application terminated.")