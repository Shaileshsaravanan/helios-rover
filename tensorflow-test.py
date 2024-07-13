import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")

# Load label map
with open("mscoco_label_map.pbtxt") as f:
    label_map = f.read()

# Function to detect objects
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = model(input_tensor)

    return detections

# Function to draw bounding boxes
def draw_boxes(frame, detections):
    height, width, _ = frame.shape

    for i in range(detections['detection_boxes'].shape[1]):
        confidence = detections['detection_scores'][0, i].numpy()
        if confidence > 0.5:
            box = detections['detection_boxes'][0, i].numpy()
            ymin, xmin, ymax, xmax = box

            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            label_index = int(detections['detection_classes'][0, i].numpy())
            label = label_map[label_index]

            if label == 'potted plant':
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Draw label and confidence
                text = f'{label}: {confidence:.2f}'
                cv2.putText(frame, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start video capture
cap = cv2.VideoCapture(0)

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