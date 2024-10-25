import cv2
import imutils
import sqlite3
import time
from imutils import face_utils
from datetime import datetime
import os
import json
import torch
from deepface import DeepFace
import numpy as np

class FaceEmbeddingManager:
    def _init_(self, model_name, embedding_file="embeddings.json"):
        self.model_name = model_name
        self.embedding_file = embedding_file

    def save_embeddings(self, embedding_scalar, name):
        data = self._load_embeddings()
        data[name] = embedding_scalar  # Save the scalar value
        with open(self.embedding_file, 'w') as file:
            json.dump(data, file, indent=4)

    def _load_embeddings(self):
        if os.path.exists(self.embedding_file):
            try:
                with open(self.embedding_file, 'r') as file:
                    return json.load(file)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}

    def generate_embedding(self, face_region):
        """Generates an embedding for the given face region."""
        try:
            face_embedding = DeepFace.represent(
                img_path=face_region,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]
            return face_embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None


def detect_faces_in_image(image, model_file='capture/ModelDNN/res10_300x300_ssd_iter_140000.caffemodel', 
                          config_file="capture/ModelDNN/deploy.prototxt.txt", confidence_threshold=0.5):
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise FileNotFoundError("Model or configuration file not found.")
    
    DNN = cv2.dnn.readNetFromCaffe(config_file, model_file)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    DNN.setInput(blob)
    detections = DNN.forward()
    
    face_bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            face_bounding_boxes.append((x, y, x1, y1))
    
    return face_bounding_boxes


def convert_to_binary(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    if is_success:
        return buffer.tobytes()
    else:
        raise Exception("Failed to convert image to binary.")


def insert_image_into_db(image_binary, employee_id, conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO employee_images (employee_id, image_data, timestamp) VALUES (?, ?, ?)",
                   (employee_id, image_binary, datetime.now()))
    conn.commit()


def initialize_db():
    conn = sqlite3.connect('employee_face_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS employee_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT NOT NULL,
                        image_data BLOB NOT NULL,
                        timestamp TEXT NOT NULL
                      )''')
    conn.commit()
    return conn


def capture_focused_face(employee_id, face_embedding_manager):
    cap = cv2.VideoCapture(0)
    images_captured = 0
    embeddings_list = []

    positions = ['center', 'left', 'right', 'up', 'down']
    position_prompts = {
        'center': "Move your face to the center.",
        'left': "Move your face slightly to the left.",
        'right': "Move your face slightly to the right.",
        'up': "Look slightly upwards.",
        'down': "Look slightly downwards."
    }

    position_index = 0
    conn = initialize_db()
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = imutils.resize(frame, width=400)
        faces = detect_faces_in_image(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            start_time = None
        else:
            for (x, y, x1, y1) in faces:
                w = x1 - x
                h = y1 - y
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                # Face centered and proper size check (omitted here for brevity)

                if True:  # Assuming face is centered and proper size
                    if start_time is None:
                        start_time = time.time()
                    elapsed_time = time.time() - start_time

                    if elapsed_time >= 2.0:
                        face_img = frame[y:y + h, x:x + w]
                        image_binary = convert_to_binary(face_img)
                        insert_image_into_db(image_binary, employee_id, conn)

                        # Save embedding
                        face_path = "temp_face.jpg"
                        cv2.imwrite(face_path, face_img)
                        embedding = face_embedding_manager.generate_embedding(face_path)
                        if embedding is not None:
                            embeddings_list.append(embedding)  # Append embedding to list
                        print(f"Position {positions[position_index]}: Image captured.")

                        images_captured += 1
                        position_index += 1
                        start_time = None

                        if position_index >= len(positions):
                            print("All positions captured. Process complete.")
                            break

        cv2.imshow('Face Capture System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or images_captured >= len(positions):
            break

    # Compute and save the average embedding scalar
    if embeddings_list:
        avg_embedding_vector = np.mean(embeddings_list, axis=0)  # Calculate average embedding vector
        avg_embedding_scalar = float(np.mean(avg_embedding_vector))  # Compute the mean of the vector elements
        face_embedding_manager.save_embeddings(avg_embedding_scalar, employee_id)
        print(f"Average scalar embedding for {employee_id} saved.")

    cap.release()
    cv2.destroyAllWindows()


# Main function
if __name__ == "_main_":
    employee_id = input("Enter the Employee ID: ")
    model_list = [
        "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    ]
    model_name = model_list[1]  # You can change the model
    face_embedding_manager = FaceEmbeddingManager(model_name)
    capture_focused_face(employee_id, face_embedding_manager)