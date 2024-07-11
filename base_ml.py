import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

MOBILE_NET_INPUT_WIDTH = 224
MOBILE_NET_INPUT_HEIGHT = 224
CLASS_NAME = "Object" 

base64_images = []  
trainingDataInputs = []
trainingDataOutputs = []
model = None

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

def image_to_base64(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_str):
    decoded = base64.b64decode(base64_str)
    buffer = BytesIO(decoded)
    return np.array(Image.open(buffer))

def gather_data():
    global base64_images

    print("Press 'c' to capture data and 'q' to quit and start prediction.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            base64_image = image_to_base64(frame)
            base64_images.append(base64_image)
            print(f"Captured image for {CLASS_NAME}")

        elif key == ord('q'):
            print("Quitting data collection.")
            break

    cap.release()
    cv2.destroyAllWindows()

def prepare_training_data():
    global trainingDataInputs, trainingDataOutputs

    print("Preparing training data...")

    for base64_image in base64_images:
        image = base64_to_image(base64_image)
        resized_frame = cv2.resize(image, (MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT))
        normalized_frame = resized_frame / 255.0
        input_tensor = tf.convert_to_tensor([normalized_frame], dtype=tf.float32)
        features = mobilenet.predict(input_tensor)
        trainingDataInputs.append(features[0])
        trainingDataOutputs.append(1)  

    for _ in range(100): 
        frame = np.random.rand(MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3)  # Dummy frame
        resized_frame = cv2.resize(frame, (MOBILE_NET_INPUT_WIDTH, MOBILE_NET_INPUT_HEIGHT))
        normalized_frame = resized_frame / 255.0
        input_tensor = tf.convert_to_tensor([normalized_frame], dtype=tf.float32)
        features = mobilenet.predict(input_tensor)
        trainingDataInputs.append(features[0])
        trainingDataOutputs.append(0) 

    trainingDataInputs = np.array(trainingDataInputs)
    trainingDataOutputs = np.array(trainingDataOutputs)
    print(f"Data preparation complete with {len(trainingDataInputs)} samples.")

def train_and_predict():
    global model
    if not mobilenet:
        print("Load the MobileNet model first!")
        return

    print("Training the model...")

    prepare_training_data()
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(trainingDataInputs.shape[1],)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(trainingDataInputs, trainingDataOutputs, epochs=10)

    print("Model trained. Ready for predictions.")
    start_predicting(model)

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
        prediction_prob = predictions[0][0]
        prediction_label = "Object Present" if prediction_prob > 0.5 else "No Object"

        print(f"Prediction Probability: {prediction_prob:.2f} - {prediction_label}")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

gather_data()
train_and_predict()