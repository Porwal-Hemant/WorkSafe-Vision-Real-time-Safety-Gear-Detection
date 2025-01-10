from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Load the video or camera feed
video_path = "../Videos/ppe-3.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if the video/camera feed is valid
if not cap.isOpened():
    print("Error: Could not open video file or camera.")
    exit()

# Set resolution (if applicable)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Load YOLO model
model_path = "best.pt"  # Path to the trained model
model = YOLO(model_path)

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 
              'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle']

# Initialize variables for FPS calculation
prev_frame_time = 0
fps = 0

while True:
    success, img = cap.read()

    # Break the loop if video ends or camera fails
    if not success:
        print("End of video or failed to read frame.")
        break

    # Run YOLO model on the frame
    results = model(img, stream=True)

    # Loop through detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))  # Draw bounding box

            # Confidence Score
            conf = round(box.conf[0] * 100, 2)

            # Class Name
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"

            # Draw label with confidence
            cvzone.putTextRect(img, f'{class_name} {conf}%', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate FPS
    new_frame_time = time.time()
    if prev_frame_time > 0:
        fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time

    # Display FPS on the frame
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

    # Show the frame
    cv2.imshow("Detection", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
