import cv2
import os
import numpy as np

# Path to your images
image_folder = './cropped_frames/'
output_folder = './detected_fish/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Loop through images in the folder
for filename in sorted(os.listdir(image_folder)):
    # Read the image
    image_path = os.path.join(image_folder, filename)
    frame = cv2.imread(image_path)

    # Apply background subtraction
    fg_mask = backSub.apply(frame)

    # Threshold the mask to get binary image
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the result
    output_filename = os.path.join(output_folder, f'detected_{filename}')
    cv2.imwrite(output_filename, frame)

    # Optionally show the result
    cv2.imshow('Detected Fish', frame)
    cv2.waitKey(1)  # Wait for 1 ms between frames

cv2.destroyAllWindows()