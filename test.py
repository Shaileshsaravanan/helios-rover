import tensorflow as tf
import numpy as np
import cv2

# Constants
MOBILE_NET_INPUT_WIDTH = 224
MOBILE_NET_INPUT_HEIGHT = 224
CLASS_NAMES = ["Class 1", "Class 2"]
STOP_DATA_GATHER = -1

# Variables
gatherDataState = 0
trainingDataInputs = []
trainingDataOutputs = []
examplesCount = [0] * len(CLASS_NAMES)
model = None

# Load MobileNet model
def load_mobile_net_feature_model():
    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=(MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
    print("MobileNet v3 loaded successfully!")
    return mobilenet

mobilenet = load_mobile_net_feature_model()

# Gather data from webcam
def gather_data():
    global gatherDataState

    print("Press 'c' to capture data, 'r' to switch class, and 'q' to quit and start prediction.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            resized_frame = cv2.resize(frame, (MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT))
            normalized_frame = resized_frame / 255.0
            input_tensor = tf.convert_to_tensor([normalized_frame], dtype=tf.float32)
            features = mobilenet.predict(input_tensor)
            trainingDataInputs.append(features[0])
            trainingDataOutputs.append(gatherDataState)
            examplesCount[gatherDataState] += 1
            print(f"Captured data for class {CLASS_NAMES[gatherDataState]} ({examplesCount[gatherDataState]} examples)")

        elif key == ord('r'):
            gatherDataState = (gatherDataState + 1) % len(CLASS_NAMES)
            print(f"Switched to class {CLASS_NAMES[gatherDataState]}")

        elif key == ord('q'):
            print("Quitting data collection.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Train the model
def train_and_predict():
    global model
    if not mobilenet:
        print("Load the MobileNet model first!")
        return

    print("Training the model...")
    xs = np.array(trainingDataInputs)
    ys = tf.one_hot(trainingDataOutputs, len(CLASS_NAMES))

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(xs.shape[1],)),
        tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(xs, ys, epochs=10)

    print("Model trained. Ready for predictions.")
    start_predicting(model)

# Start predicting
def start_predicting(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT))
        normalized_frame = resized_frame / 255.0
        input_tensor = tf.convert_to_tensor([normalized_frame], dtype=tf.float32)
        features = mobilenet.predict(input_tensor)
        predictions = model.predict(features)
        top_prediction = np.argmax(predictions)
        print(f"Prediction: {CLASS_NAMES[top_prediction]}")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Reset the data
def reset():
    global trainingDataInputs, trainingDataOutputs, examplesCount, gatherDataState
    print("Resetting...")
    trainingDataInputs = []
    trainingDataOutputs = []
    examplesCount = [0] * len(CLASS_NAMES)
    gatherDataState = 0
    print("Reset complete. Collect new data.")

# Example usage:
# Gather data from webcam
gather_data()

# Train the model and start predictions
train_and_predict()

# Reset the data
reset()