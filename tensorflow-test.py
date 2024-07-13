import cv2
import torch

# Load the YOLOv5 model (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

# Define the confidence threshold
confidence_threshold = 0.5

# Loop for capturing frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Extract results as a pandas DataFrame
    detections = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame

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

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()