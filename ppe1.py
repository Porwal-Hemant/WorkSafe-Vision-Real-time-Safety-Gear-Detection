from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import threading  # For playing and stopping alert sound
import os
# from playsound import playsound  # Import for playing sound

# Function to play alert sound in a loop
# def play_alert():
    # while alert_playing:
    #     playsound("alert.mp3", block=False)

# Choose video or camera
choice = input("Enter '1' for Camera or '2' for Video: ")

if choice == '1':
    cap = cv2.VideoCapture(0)  # Open webcam
    cap.set(3, 640)  # Set frame width
    cap.set(4, 480)  # Set frame height
elif choice == '2':
    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit()
    cap = cv2.VideoCapture(video_path)  # Open video file
else:
    print("Invalid choice! Exiting...")
    exit()

# Load the YOLO model
model = YOLO("best.pt")

# Define class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

prev_frame_time = 0
new_frame_time = 0
fps = 0  # Initialize fps variable
alert_playing = False  # Flag to check if the alert sound is playing
alert_thread = None  # Thread for playing the alert sound

# Check if camera or video is opened
if not cap.isOpened():
    print("Error: Camera/Video not found or could not be opened.")
    exit()

try:
    while True:
        success, img = cap.read()

        if not success:
            print("Failed to capture image or end of video.")
            break

        results = model(img, stream=True)
        alert_triggered = False  # Flag to check if alert needs to be triggered

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Trigger alert if necessary
                if classNames[cls] in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    alert_triggered = True

        # # Handle alert sound************
        # if alert_triggered and not alert_playing:
        #     alert_playing = True
        #     alert_thread = threading.Thread(target=play_alert)
        #     alert_thread.start()
        # elif not alert_triggered and alert_playing:
        #     alert_playing = False
        #     if alert_thread:
        #         alert_thread.join()  # Wait for the thread to finish

        # Calculate FPS
        new_frame_time = time.time()
        if prev_frame_time != 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            fps = int(fps)  # Convert to integer
        prev_frame_time = new_frame_time

        # Display FPS on the image
        cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

        print(f'FPS: {fps}')
        
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break
finally:
    alert_playing = False  # Stop the alert sound
    if alert_thread:
        alert_thread.join()  # Ensure the alert sound thread stops
    cap.release()  # Release the camera or video when done
    cv2.destroyAllWindows()  # Close all OpenCV windows



"""

This code is for real-time object detection using a YOLO model to detect whether workers are wearing Personal Protective Equipment (PPE) (like hardhats, masks, or safety vests) in a video stream or from a webcam.

Key Functionality:
Video/Camera Input: The program asks the user whether to use the webcam (camera) or a video file. It then loads the appropriate video source.

YOLO Model: The code uses a YOLOv8 model (best.pt) to detect objects in the video feed. It specifically looks for objects like hardhats, masks, safety vests, and more.

Bounding Boxes & Labels: For each detected object, the program draws a bounding box around it and displays the confidence level and class name (e.g., "NO-Hardhat" or "Person").

Alert Mechanism (commented out): The code is designed to trigger an alert if a person is detected without the necessary PPE (like a hardhat, mask, or safety vest). The alert would play a sound, but this feature is currently commented out.

FPS Calculation: The program calculates and displays the frames per second (FPS) to show how smoothly the detection is happening.

Exit Condition: The program keeps running until the user presses the "q" key, after which it releases the camera or video file and closes the windows.

Key Points:
The video or camera feed is processed frame by frame.
The YOLO model is used for real-time object detection, specifically looking for missing PPE (hardhats, masks, etc.).
The program calculates FPS to monitor performance.
An alert system is in place to trigger a sound when certain conditions are met (though it's currently disabled).

"""