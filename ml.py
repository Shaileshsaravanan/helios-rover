import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2 as cv
import time
import json

# Sample data: list of base64 images
data = []

# Image parameters
img_height, img_width = 224, 224

# Function to decode base64 image
def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image = image.resize((img_height, img_width))
    return np.array(image)

# Function to encode image to base64
def encode_base64_image(image):
    _, buffer = cv.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

# Prepare the dataset
images = [decode_base64_image(base64_image) for base64_image in data]
labels = np.ones(len(images))  # All images belong to the single class

images = np.array(images) / 255.0
labels = tf.keras.utils.to_categorical(labels, num_classes=2)  # Two classes: object present or not

# Split the data into training and validation sets
split_index = int(0.8 * len(images))
train_images, train_labels = images[:split_index], labels[:split_index]
val_images, val_labels = images[split_index:], labels[split_index:]

# Define the model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)

# Save the model
model_save_path = 'model.h5'
model.save(model_save_path)

# Load the model
model = tf.keras.models.load_model(model_save_path)

# Open a connection to the webcam
cap = cv.VideoCapture(0)

# Variables to store the last seen object and time
last_seen_object = False
last_seen_time = 0

# List to store detections
detections = []

def check_and_print(predictions, frame):
    global last_seen_object, last_seen_time, detections

    class_confidence = predictions[0][1]  # Confidence for the class

    if class_confidence > 0.5:
        current_time = time.time()
        if not last_seen_object or (current_time - last_seen_time) > 5:  # 5 seconds interval
            print(f"Object detected with confidence: {class_confidence * 100:.2f}%")
            last_seen_object = True
            last_seen_time = current_time

            # Encode the frame as base64
            encoded_image = encode_base64_image(frame)

            # Store detection
            detection = {
                'image': encoded_image,
                'confidence': class_confidence * 100
            }
            detections.append(detection)
    else:
        if last_seen_object:
            print("Object not detected")
            last_seen_object = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv.resize(frame, (img_height, img_width))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the class of the frame
    predictions = model.predict(img)

    # Check and print the result
    check_and_print(predictions, frame)

    # Show the webcam feed
    cv.imshow("Webcam Feed", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()

# Save detections to a JSON file
with open('detections.json', 'w') as f:
    json.dump(detections, f)

print(f"Detections saved to detections.json")