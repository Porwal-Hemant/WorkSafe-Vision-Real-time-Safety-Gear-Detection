# ## Video without alert sound


# from ultralytics import YOLO
# import cv2
# import cvzone
# import math
# import time

# cap = cv2.VideoCapture(0)  # Change camera index if necessary
# cap.set(3, 1280)  # Set frame width
# cap.set(4, 720)   # Set frame height
# # cap.set(3, 640)
# # cap.set(4, 480)

# # cap = cv2.VideoCapture("../Videos/bikes.mp4") 
# # cap = cv2.VideoCapture("../Videos/people.mp4") 

# cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video

# model = YOLO("../Yolo-Weights/yolov8l.pt")
# # model = YOLO("../Yolo-Weights/yolov8n.pt")
# model = YOLO("best.pt")

# classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
#               'Safety Vest', 'machinery', 'vehicle']

# prev_frame_time = 0
# new_frame_time = 0
# fps = 0  # Initialize fps variable

# # Check if camera is opened
# if not cap.isOpened():
#     print("Error: Camera not found or could not be opened.")
#     exit()

# while True:
#     sucess, img = cap.read()

#     if not sucess:
#         print("Failed to capture image")
#         break

#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])

#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

#     # Calculate FPS
#     new_frame_time = time.time()
#     if prev_frame_time != 0:
#         fps = 1 / (new_frame_time - prev_frame_time)
#         fps = int(fps)  # Convert to integer
#     prev_frame_time = new_frame_time

#     # Display FPS on the image
#     cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)

#     print(f'FPS: {fps}')
    
#     cv2.imshow("Image", img)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
#         break

# cap.release()  # Release the camera when done
# cv2.destroyAllWindows()  # Close all OpenCV windows


# # The code captures a video stream (either from a camera or video file), applies YOLO object detection to identify predefined classes (like Hardhat, Mask, Person, etc.), draws bounding boxes around detected objects, displays the class name and confidence score, and shows the FPS. This is used in applications like PPE (Personal Protective Equipment) compliance monitoring, where detecting whether workers are wearing safety gear (hardhats, masks, etc.) is important.




from ultralytics import YOLO
import cv2
import cvzone
import math
import time


cap = cv2.VideoCapture("./Videos/ppe-3.mp4")  #
cap.set(3, 1280)  
cap.set(4, 720)   


model = YOLO("best.pt")  


classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
              'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle' , 'goggles']

prev_frame_time = 0
fps = 0  

if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("End of video or failed to capture image")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
           
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            cls = int(box.cls[0])
           
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                               (max(0, x1), max(35, y1)), scale=1, thickness=1)

# putTextRect method is the function which can help u to plot rectangle across the detected coordinates
    
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (new_frame_time - prev_frame_time)
        fps = int(fps)  
    prev_frame_time = new_frame_time
    
    cvzone.putTextRect(img, f'FPS: {fps}', (10, 50), scale=1, thickness=1)
    print(f'FPS: {fps}')
    
    
    cv2.imshow("Image", img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()  



# issue at the end it is not destroying all windows 




