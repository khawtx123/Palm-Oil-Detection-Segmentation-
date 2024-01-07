import os
import cv2
import numpy as np

# Function to read images from directory, convert to grayscale, and write back
def process_images(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions as needed
            img_path = os.path.join(directory_path, filename)

            # Read image in grayscale mode
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            unique_values = np.unique(img)
            cv2.imshow('img',img)
            cv2.waitKey(0)
            # Print unique values
            print("Unique values in the mask:", unique_values)
            _, thresholded_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            cv2.imwrite(img_path, thresholded_mask)

mask_dir = r"C:\Users\USER\Downloads\rodent\SegmentationClass"
# Process images in the directory
process_images(mask_dir)

print("Image processing completed.")


