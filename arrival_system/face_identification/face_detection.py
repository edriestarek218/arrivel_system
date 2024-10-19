import cv2
import numpy as np
import os

def detect_faces_in_image(image, output_dir="captured_faces", 
                          model_file="capture\ModelDNN\\res10_300x300_ssd_iter_140000.caffemodel", 
                          config_file="capture\ModelDNN\deploy.prototxt.txt", confidence_threshold=0.5):
    """
    Detects faces in a given image using a DNN model, saves cropped face images, and returns bounding boxes.
    
    Parameters:
    - image: The input image (as a NumPy array) where faces will be detected.
    - output_dir: Directory to save the cropped face images.
    - model_file: The path to the pre-trained face detection model.
    - config_file: The path to the model's configuration file.
    - confidence_threshold: The confidence threshold for detecting faces.
    
    Returns:
    - A list of bounding boxes [(x1, y1, x2, y2), ...] for each detected face.
    """
    # Check if the model and config files exist
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        raise FileNotFoundError("Model or configuration file not found. Please check the file paths.")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
            
            # Crop the face from the image
            face = image[y:y1, x:x1]
            
            # Save the cropped face image
            face_file_name = os.path.join(output_dir, f"face_{i}.png")
            cv2.imwrite(face_file_name, face)
    
    # Return the list of face bounding boxes
    return face_bounding_boxes
