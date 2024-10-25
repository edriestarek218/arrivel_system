import cv2
import os
import csv
from deepface import DeepFace
from datetime import datetime 
from face_identification.face_detection import detect_faces_in_image

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
model_name = models[1]  # Facenet in this case

# To keep track of recognized names
recognized_names = set()

def save_data(name, arrival_time, csv_attend='attend.csv'):
    with open(csv_attend, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, arrival_time])

def recognize_faces(frame, db_path="face_identification/DataBase", csv_file='attend.csv'):
    """
    Function to detect and recognize faces in a frame.
    
    Parameters:
    - frame: The video frame in which to detect and recognize faces.
    - db_path: The path to the face database for DeepFace recognition.
    
    Returns:
    - frame: The frame with recognized faces labeled and bounding boxes drawn.
    """
    # Detect faces in the frame
    faces = detect_faces_in_image(frame)
    
    # Loop through each detected face
    for (x, y, x1, y1) in faces:
        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        # Extract the face region from the frame
        face_region = frame[y:y1, x:x1]

        # Check if face_region has valid dimensions
        if face_region.shape[0] > 0 and face_region.shape[1] > 0:
            try:
                # Pass the face region (array) directly to DeepFace
                dfs = DeepFace.find(
                    img_path=face_region,
                    db_path=db_path,
                    model_name=model_name,
                    enforce_detection=False  # Skip detection as face is already detected
                )
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if len(dfs) > 0 and len(dfs[0]) > 0:
                    # Extract the identity of the closest match (file name)
                    identity = dfs[0].iloc[0]["identity"]  # Extract the identity (full path)
                    
                    # Extract just the name from the file path
                    identity_name = os.path.basename(identity).split(".")[0]
                    
                    print(f"Identified person: {identity_name}")
                    
                    
                    
                    
                    
                    
                    
                    

                    #check for duplicate name 
                    if identity_name not in recognized_names:
                        recognized_names.add(identity_name)  
                        save_data(identity_name, current_time, csv_file)  

                    # Draw the name of the person inside the bounding box
                    cv2.putText(frame, identity_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    
                    
                    
                    
                    

                    # Save "Unknown" only if it hasn't been recorded yet
                    if "Unknown" not in recognized_names:
                        recognized_names.add("Unknown")
                        save_data("Unknown", current_time, csv_file)
                    
            except Exception as e:
                print(f"Error: {e}")

    # Return the frame with labeled faces
    return frame
