import cv2
from ultralyticsplus import YOLO

# Load model
model = YOLO('foduucom/plant-leaf-detection-and-classification')

# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # Maximum number of detections per image

# Open a connection to the camera (0 is the default camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Perform inference
    results = model.predict(frame)

    # Draw the hitboxes on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
        conf = box.conf[0]  # Extract confidence score
        label = f'Plant ({conf:.2f})'  # Label with confidence score

        # Draw rectangle around the detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put text above the detected object
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Plant Leaf Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()