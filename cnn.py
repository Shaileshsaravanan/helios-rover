import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Function to split dataset into training and validation sets
def split_dataset(source_dir, train_dir, validation_dir, split_size):
    all_files = os.listdir(source_dir)
    np.random.shuffle(all_files)
    train_files, validation_files = train_test_split(all_files, test_size=split_size)
    
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    for file in validation_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(validation_dir, file))

# Define directories
source_dir = 'dataset/leaves'
train_dir = 'dataset/train/plant'
validation_dir = 'dataset/validation/plant'

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Split dataset
split_dataset(source_dir, train_dir, validation_dir, split_size=0.2)

# Function to create "No Plant" images by blacking out
def create_no_plant_images(source_dir, dest_dir, num_images):
    os.makedirs(dest_dir, exist_ok=True)
    all_files = os.listdir(source_dir)
    
    for i in range(num_images):
        img_path = os.path.join(source_dir, np.random.choice(all_files))
        img = cv2.imread(img_path)
        
        # Create a black image with the same dimensions
        black_img = np.zeros_like(img)
        
        # Save the black image
        cv2.imwrite(os.path.join(dest_dir, f'no_plant_{i}.jpg'), black_img)

# Define directories for "No Plant" images
no_plant_train_dir = 'dataset/train/no_plant'
no_plant_validation_dir = 'dataset/validation/no_plant'

# Create "No Plant" images
create_no_plant_images(train_dir, no_plant_train_dir, num_images=100)
create_no_plant_images(validation_dir, no_plant_validation_dir, num_images=20)

# Data preprocessing with ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build and compile the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50
)

# Save the model
model.save('plant_detection_model.h5')
