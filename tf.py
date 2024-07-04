import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')  # or Dense(num_classes, activation='softmax') for multiple classes
    ])
    model.compile(loss='binary_crossentropy',  # or 'categorical_crossentropy' for multiple classes
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def train_model(model, train_dir, validation_dir, epochs=15):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'  # or 'categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'  # or 'categorical'
    )

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )
    model.save('model.h5')

# Paths to your training and validation directories
train_dir = 'path/to/train'
validation_dir = 'path/to/validation'

# Build and train the initial model
model = build_model()
train_model(model, train_dir, validation_dir)