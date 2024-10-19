import cv2
from face_identification.identification import recognize_faces
from capture.focusV2 import capture_focused_face

def face_recognition():
    """Function to handle real-time face recognition"""
    # Define the RTSP URL or video source
    rtsp_url = 0  # Use 0 for the default webcam, or replace with RTSP URL
    
    # Open the video stream
    cap = cv2.VideoCapture(rtsp_url)
    
    # Check if the video stream is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Start video capture and face recognition
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Call the recognize_faces function to detect and recognize faces
        frame = recognize_faces(frame)
        
        # Display the resulting frame with face recognition
        cv2.imshow('Real-time face recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to choose between face recognition or capture"""
    print("Select the operation:")
    print("1. Real-time Face Recognition")
    print("2. Capture Focused Face for Employee")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        face_recognition()  # Run face recognition
    elif choice == '2':
        employee_id = input("Enter the Employee ID: ").strip()
        capture_focused_face(employee_id)  # Run focused face capture
    else:
        print("Invalid choice. Please restart and choose either 1 or 2.")

# Run the main function
if __name__ == "__main__":
    main()

