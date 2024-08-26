import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='helios-rover/dataset/dataset.yaml', epochs=100, imgsz=640)
model.save('plant_detection.pt')  

results = model.val()
print(results)

def predict_live_stream(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for result in results:
            for box in result['boxes']:
                x1, y1, x2, y2 = box['xyxy']
                confidence = box['conf']
                class_id = box['class']

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Live Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_live_stream(model)