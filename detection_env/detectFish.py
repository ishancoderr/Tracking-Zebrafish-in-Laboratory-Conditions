import cv2
import numpy as np
import os

# Paths for input and output folders
input_folder = './new_cropped_frames/'  # Folder with 720 cropped images
output_folder = './moving_objects/'  # Output folder for images with detected moving objects

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all images in the input folder
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

# Load images to calculate the median background model
images = []
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    if image is not None:
        images.append(image)

# Calculate the median image to estimate the background
median_background = np.median(images, axis=0).astype(dtype=np.uint8)

# Process each image to detect moving objects
for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, f"moving_{image_file}")
    
    # Read the current image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not open image {input_path}.")
        continue

    # Compute the absolute difference between the current image and the median background
    diff = cv2.absdiff(image, median_background)

    # Convert the difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale difference to create a binary mask
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours of the detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum contour area to filter noise
            x, y, w, h = cv2.boundingRect(contour)
            # Crop the region of interest (moving object) and save it
            moving_object = image[y:y+h, x:x+w]
            moving_object_filename = os.path.join(output_folder, f"moving_object_{image_file}")
            cv2.imwrite(moving_object_filename, moving_object)
            print(f"Saved moving object: {moving_object_filename}")

print("Moving object detection and saving complete.")