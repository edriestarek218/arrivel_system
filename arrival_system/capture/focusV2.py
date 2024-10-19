import cv2
import imutils
import sqlite3
import time
from imutils import face_utils
from datetime import datetime


import cv2
import numpy as np
import os

def detect_faces_in_image(image, model_file="capture\ModelDNN\\res10_300x300_ssd_iter_140000.caffemodel", 
                          config_file="capture\ModelDNN\deploy.prototxt.txt", confidence_threshold=0.5):
    """
    Detects faces in a given image using a DNN model and returns the bounding boxes of detected faces.
    
    Parameters:
    - image: The input image (as a NumPy array) where faces will be detected.
    - model_file: The path to the pre-trained face detection model.
    - config_file: The path to the model's configuration file.
    - confidence_threshold: The confidence threshold for detecting faces.
    
    Returns:
    - A list of bounding boxes [(x1, y1, x2, y2), ...] for each detected face.
    """
    # Check if the model and config files exist
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise FileNotFoundError("Model or configuration file not found. Please check the file paths.")
    
    # Load the DNN model
    DNN = cv2.dnn.readNetFromCaffe(config_file, model_file)
    
    # Preprocess the image for the DNN model
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    
    # Pass the blob through the network to detect faces
    DNN.setInput(blob)
    detections = DNN.forward()
    
    # List to store the bounding boxes of detected faces
    face_bounding_boxes = []
    
    # Iterate over all detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only proceed if confidence is above the threshold
        if confidence > confidence_threshold:
            # Get the bounding box for the face detection
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            face_bounding_boxes.append((x, y, x1, y1))
    
    # Return the list of face bounding boxes
    return face_bounding_boxes


import cv2
import imutils
import sqlite3
import time
from imutils import face_utils
from datetime import datetime

# Function to convert the image to binary for database storage
def convert_to_binary(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    if is_success:
        return buffer.tobytes()
    else:
        raise Exception("Failed to convert image to binary.")


# Function to insert image data into the SQLite database
def insert_image_into_db(image_binary, employee_id, conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO employee_images (employee_id, image_data, timestamp) VALUES (?, ?, ?)",
                   (employee_id, image_binary, datetime.now()))
    conn.commit()


# Function to initialize the database and create the required table
def initialize_db():
    conn = sqlite3.connect('employee_face_data.db')  # Create or connect to the SQLite DB
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS employee_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        employee_id TEXT NOT NULL,
                        image_data BLOB NOT NULL,
                        timestamp TEXT NOT NULL
                      )''')
    conn.commit()
    return conn


# Function to check face alignment and positioning
def is_face_centered(face, frame):
    (x, y, w, h) = face
    frame_height, frame_width = frame.shape[:2]

    face_center_x = x + w // 2
    face_center_y = y + h // 2

    region_x_min = frame_width * 0.3
    region_x_max = frame_width * 0.7
    region_y_min = frame_height * 0.3
    region_y_max = frame_height * 0.7

    return region_x_min < face_center_x < region_x_max and region_y_min < face_center_y < region_y_max


# Function to ensure face size is within a desired range
def is_face_proper_size(face, frame):
    (x, y, w, h) = face
    frame_height, frame_width = frame.shape[:2]

    face_area = w * h
    frame_area = frame_height * frame_width
    face_ratio = face_area / frame_area

    return 0.1 < face_ratio < 0.5


# Function to capture face images at 5 positions with a 1-second hold
def capture_focused_face(employee_id):
    cap = cv2.VideoCapture(0)
    images_captured = 0  # Counter for images captured

    positions = ['center', 'left', 'right', 'up', 'down']
    position_prompts = {
        'center': "Move your face to the center.",
        'left': "Move your face slightly to the left.",
        'right': "Move your face slightly to the right.",
        'up': "Look slightly upwards.",
        'down': "Look slightly downwards."
    }

    position_index = 0  # Start with the first position (center)
    conn = initialize_db()

    start_time = None  # To track when the face was first aligned correctly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame = imutils.resize(frame, width=400)
        
        # Detect faces in the frame using your custom face detection function
        faces = detect_faces_in_image(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            start_time = None  # Reset timer if no face is detected
        else:
            for (x, y, x1, y1) in faces:
                w = x1 - x
                h = y1 - y
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

                centered = is_face_centered((x, y, w, h), frame)
                proper_size = is_face_proper_size((x, y, w, h), frame)

                # Only proceed if face is correctly centered and has proper size for current position
                if centered and proper_size:
                    if start_time is None:
                        start_time = time.time()  # Start the 1-second timer
                    elapsed_time = time.time() - start_time

                    cv2.putText(frame, f"Face {positions[position_index]} OK! Hold still...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Check if face is held in position for 1 second
                    if elapsed_time >= 2.0:
                        # Capture the image for the current position
                        face_img = frame[y:y + h, x:x + w]
                        image_binary = convert_to_binary(face_img)
                        insert_image_into_db(image_binary, employee_id, conn)
                        print(f"Position {positions[position_index]}: Image captured and saved.")

                        images_captured += 1
                        position_index += 1  # Move to the next position
                        start_time = None  # Reset timer for the next position

                        if position_index >= len(positions):
                            print("All 5 positions captured. Confirmation complete.")
                            break
                else:
                    # Provide instructions for the current position
                    cv2.putText(frame, position_prompts[positions[position_index]], (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    start_time = None  # Reset timer if face is not aligned properly

                # Draw the ROI where the face should be aligned
                frame_height, frame_width = frame.shape[:2]
                roi_x_min = int(frame_width * 0.3)
                roi_x_max = int(frame_width * 0.7)
                roi_y_min = int(frame_height * 0.3)
                roi_y_max = int(frame_height * 0.7)
                cv2.rectangle(frame, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (255, 255, 0), 2)

        cv2.imshow('Face Capture System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or images_captured >= len(positions):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the face capture system
if __name__ == "__main__":
    employee_id = input("Enter the Employee ID: ")
    capture_focused_face(employee_id)
