import cv2
import os
import csv
from deepface import DeepFace
from face_identification.face_detection import detect_faces_in_image
from datetime import datetime 
import json
import torch

# # Check for PyTorch GPU usage
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     print("Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# Model list
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]

# Define the recognition model you want to use
model_name = models[1]  # VGG-Face in this case

# Function to recognize faces
def save_embeddings(embeddings, name, embedding_file="face_identification/embeddings.json"):
    if os.path.exists(embedding_file):
        try:
            with open(embedding_file, 'r') as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    else:
        data = {}

    data[name] = embeddings
    with open(embedding_file, 'w') as file:
        json.dump(data, file, indent=4)

# Modified recognize_faces function to include embedding saving
def recognize_faces(frame, db_path="face_identification/DataBase", attendance_file="face_identification/attendance.csv", embedding_file="face_identification/embeddings.json"):
    unknown_dir = os.path.join(db_path, "unknown_faces")
    if not os.path.exists(unknown_dir):
        os.makedirs(unknown_dir)
    if not os.path.exists(attendance_file):
        with open(attendance_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Person_ID', 'Timestamp'])

    faces = detect_faces_in_image(frame)
    
    for (x, y, x1, y1) in faces:
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        face_region = frame[y:y1, x:x1]

        if face_region.shape[0] > 0 and face_region.shape[1] > 0:
            try:
                # Specify PyTorch backend
                device = torch.device("cuda:1")
                deep = DeepFace().to(device)
                dfs = deep.find(
                    img_path=face_region,  # Pass the face region as a NumPy array
                    db_path=db_path,
                    model_name=model_name,
                    enforce_detection=False, # Skip detection as face is already detected
                    threshold=0.8, 
                      # Use OpenCV as detector
                    distance_metric="euclidean_l2", # Force using PyTorch backend
                )

                face_embedding = deep.represent(
                    img_path=face_region,
                    model_name=model_name,
                    enforce_detection=False,
                      # Use OpenCV as detector
                     # Use PyTorch as backend
                )[0]["embedding"]

                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if len(dfs) > 0 and len(dfs[0]) > 0:
                    identity = dfs[0].iloc[0]["identity"]
                    identity_name = os.path.basename(os.path.dirname(identity))
                    print(identity_name)

                    with open(attendance_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([identity_name, current_time])

                    cv2.putText(frame, identity_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    unknown_id = f"unknown_{len(os.listdir(unknown_dir)) + 1}"
                    unknown_dir_path = os.path.join(unknown_dir, f"{unknown_id}.jpg")
                    cv2.imwrite(unknown_dir_path, face_region)
                    save_embeddings(face_embedding, unknown_id, embedding_file)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("Unknown")

                    with open(attendance_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([unknown_id, current_time])
            except Exception as e:
                print(f"Error: {e}")

    return frame
