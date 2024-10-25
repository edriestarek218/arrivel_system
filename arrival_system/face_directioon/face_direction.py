import numpy as np
import cv2
import mediapipe as mp
import json

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=1, circle_radius=1)


def process_image(image):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            # Get 2D and 3D Coordinates
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera Matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation and translation vectors
            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            # Calculate rotation angles from rotation vector
            rmat, jac = cv2.Rodrigues(rotation_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            # Determine direction based on rotation angles
            if y_angle < -10:
                direction = "Looking Left"
            elif y_angle > 10:
                direction = "Looking Right"
            elif x_angle < -10:
                direction = "Looking Down"
            elif x_angle > 10:
                direction = "Looking Up"
            else:
                direction = "Forward"

            # Return only the direction as a JSON object
            result = {
                "direction": direction
            }

            return json.dumps(result)

    return json.dumps({"error": "No face detected"})

