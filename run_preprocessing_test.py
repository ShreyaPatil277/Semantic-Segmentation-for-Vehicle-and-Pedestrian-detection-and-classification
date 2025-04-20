from dataset_preprocessing import preprocess_data

image_dir = "Dataset/leftImg8bit/test"
mask_dir = "Dataset/gtFine/test"
image_output_dir = "preprocessed_images/test"
mask_output_dir = "preprocessed_masks/test"

preprocess_data(image_dir, mask_dir, image_output_dir, mask_output_dir)
