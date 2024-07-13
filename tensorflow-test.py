import cv2
import numpy as np
import torch

# Load the YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to detect objects
def detect_objects(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get predictions as a pandas DataFrame
    return detections

# Function to draw bounding boxes
def draw_boxes(frame, detections):
    height, width, _ = frame.shape

    for _, row in detections.iterrows():
        confidence = row['confidence']
        if confidence > 0.5:
            xmin = int(row['xmin'])
            xmax = int(row['xmax'])
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])

            label = row['name']
            if label == 'plant':  # This assumes 'plant' is a class in the YOLOv5 model
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Draw label and confidence
                text = f'{label}: {confidence:.2f}'
                cv2.putText(frame, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    frame = draw_boxes(frame, detections)

    cv2.imshow('Plant Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()