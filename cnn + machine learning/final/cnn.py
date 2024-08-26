import tensorflow as tf
import numpy as np
import os
import cv2
import albumentations as A
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Sequential

MODEL_SAVE_PATH = 'trained_tf_model.h5'
BATCH_SIZE = 8
IMG_SIZE = 640
EPOCHS = 20

# Define preprocessing and augmentation pipeline
transform = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE, interpolation=cv2.INTER_LINEAR, always_apply=True),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=(0, 0.2), p=0.5),
    A.OneOf([
        A.Lambda(image=lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=0.5),
    ], p=1.0),
    A.GaussNoise(var_limit=(0, 10), p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),  # Normalizing the images
])

def preprocess_image(image):
    augmented = transform(image=image)
    img = augmented['image']
    if len(img.shape) == 2:  # If the image is grayscale
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # Convert grayscale to RGB
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_function(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(image_paths, labels, batch_size):
    def gen():
        for path, label in zip(image_paths, labels):
            image, label = preprocess_function(path, label)
            # Wrap label in an array
            yield image, np.expand_dims(label, axis=0)
    
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32)  # Expecting a shape of (1,)
        )
    )
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_data_from_directory(directory):
    image_paths = []
    labels = []
    
    for img_name in os.listdir(directory):
        image_paths.append(os.path.join(directory, img_name))
        labels.append(1)  # All images are labeled as class 1 (leaves)
    
    return image_paths, labels

def train_model(train_dir, val_dir):
    train_image_paths, train_labels = load_data_from_directory(train_dir)
    val_image_paths, val_labels = load_data_from_directory(val_dir)
    
    train_dataset = create_dataset(train_image_paths, train_labels, BATCH_SIZE)
    val_dataset = create_dataset(val_image_paths, val_labels, BATCH_SIZE)

    model = build_model()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_dir = '/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/train/images'  
    val_dir = '/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/valid/images'  
    train_model(train_dir, val_dir)