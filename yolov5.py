import cv2
import torch
import base64
import numpy as np
from io import BytesIO
from PIL import Image

# Load the YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Define the confidence threshold
confidence_threshold = 0.5

# Define the size of the frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the center of the frame
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

def image_to_base64(img):
    """ Convert an OpenCV image to a base64 string. """
    _, buffer = cv2.imencode('.jpg', img)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

# Loop for capturing frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Extract results as a pandas DataFrame
    detections = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame

    # Variable to check if any detection is centered
    centered_detection = False

    # Loop through the detections
    for _, row in detections.iterrows():
        class_id = int(row['class'])
        confidence = row['confidence']
        if confidence > confidence_threshold:
            # Draw the bounding box
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f'{results.names[class_id]} {confidence:.2f}'
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Check if the bounding box is centered
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            if (frame_center_x - 50 <= box_center_x <= frame_center_x + 50 and
                frame_center_y - 50 <= box_center_y <= frame_center_y + 50):
                # Bounding box is centered
                centered_detection = True
                # Capture the image
                base64_str = image_to_base64(frame)
                # Print the base64 string
                print(base64_str)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()