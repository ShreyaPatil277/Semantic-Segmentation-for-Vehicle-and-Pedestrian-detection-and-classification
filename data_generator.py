import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator(image_dir, mask_dir, batch_size=8, img_size=(256, 256)):
    # ImageDataGenerator object to rescale images between 0 and 1
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)

    # Image generator (images will be loaded from the image directory)
    image_generator = image_datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,  # We don't need labels for images, just data
        seed=42
    )

    # Mask generator (masks will be loaded from the mask directory)
    mask_generator = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,  # Masks are not classified
        color_mode='grayscale',  # Since it's a single channel mask
        seed=42
    )

    # Combine both generators (image + mask) to return together
    while True:
        img_batch = image_generator.next()
        mask_batch = mask_generator.next()
        yield img_batch, mask_batch
