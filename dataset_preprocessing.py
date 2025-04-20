import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

IMG_SIZE = (256, 256)
AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

def preprocess_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                img = cv2.resize(img, IMG_SIZE)
                img = AUGMENT(image=img)["image"]
                img = img / 255.0
                save_path = os.path.join(output_dir, file.replace(".jpg", ".npy").replace(".png", ".npy"))
                np.save(save_path, img)


def preprocess_masks(mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".png"):
                mask_path = os.path.join(root, file)
                mask = cv2.imread(mask_path, 0)
                if mask is None:
                    print(f"Failed to read mask: {mask_path}")
                    continue
                mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
                save_path = os.path.join(output_dir, file.replace(".png", ".npy"))
                np.save(save_path, mask)

image_dir = "C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/Dataset/leftImg8bit/train"  # Images ka path
mask_dir = "C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/Dataset/gtFine/train"  # Masks ka path

image_output_dir = "C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/preprocessed_images/train"
mask_output_dir = "C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/preprocessed_masks/train"



def preprocess_data(image_dir, mask_dir, image_output_dir, mask_output_dir):
    preprocess_images(image_dir, image_output_dir)
    preprocess_masks(mask_dir, mask_output_dir)

# Call this function with your directories
# Example:
# preprocess_data('path/to/images', 'path/to/masks', 'path/to/save/images', 'path/to/save/masks')
