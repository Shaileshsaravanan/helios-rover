import tensorflow as tf
import numpy as np
import cv2
import albumentations as A

# Load the saved model
MODEL_SAVE_PATH = '/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/final/trained_tf_model.h5'
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Check model input shape
print("Model input shape:", model.input_shape)

# Update IMG_SIZE based on the model input shape
IMG_SIZE = model.input_shape[1]  # Assuming the shape is (None, IMG_SIZE, IMG_SIZE, 3)

# Define preprocessing and augmentation pipeline
transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_LINEAR, always_apply=True),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),  # Normalizing the images
])

def preprocess_image(image):
    augmented = transform(image=image)
    img = augmented['image']
    img = img.astype(np.float32) / 255.0
    return img

def main():
    # Open a video capture object
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_image(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(preprocessed_frame)
        label = (predictions[0][0] > 0.5).astype(int)  # Binary classification

        # Display the result
        label_text = 'Class 1' if label == 1 else 'Class 0'
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
