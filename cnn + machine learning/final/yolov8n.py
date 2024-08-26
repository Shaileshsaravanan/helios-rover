import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO

MODEL_SAVE_PATH = 'trained_yolo_model.pt'
DATA_CFG_PATH = 'cnn + machine learning/dataset/data.yaml'

def grayscale_transform(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

transform = A.Compose([
    A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_LINEAR, always_apply=True),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
    A.RandomCrop(height=640, width=640, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=(0, 0.2), p=0.5),
    A.OneOf([
        A.Lambda(image=grayscale_transform, p=0.15),
        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=25, val_shift_limit=0, p=0.5),
    ], p=1.0),
    A.GaussNoise(var_limit=(0, 10), p=0.5),
    ToTensorV2()
])

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(
        data=DATA_CFG_PATH,
        epochs=20,
        imgsz=640,
        batch=8,
        device='cpu',
        workers=7,
        augment=True,
        cache=False
    )
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()
