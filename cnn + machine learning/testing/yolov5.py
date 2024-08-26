import cv2
from ultralyticsplus import YOLO


model = YOLO('plant_detection_yolo.pt')
model.overrides['conf'] = 0.25  
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False  
model.overrides['max_det'] = 1000 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    results = model.predict(frame)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        conf = box.conf[0]
        label = f'Plant ({conf:.2f})'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Plant Leaf Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()