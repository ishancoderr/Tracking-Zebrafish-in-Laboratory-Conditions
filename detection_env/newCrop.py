import cv2
import os

input_folder = './cropped_frames/'
output_folder = './new_cropped_frames/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sample_image_file = 'cropped_frame_0.jpg' 
sample_image_path = os.path.join(input_folder, sample_image_file)

sample_image = cv2.imread(sample_image_path)
if sample_image is None:
    print(f"Error: Could not open image {sample_image_path}.")
    exit(1)

# Resize the sample image for easier bounding box selection
scale_percent = 50  
width = int(sample_image.shape[1] * scale_percent / 100)
height = int(sample_image.shape[0] * scale_percent / 100)
resized_sample = cv2.resize(sample_image, (width, height))

print("Select the region for the bounding box and press ENTER.")
bounding_box = cv2.selectROI("Select Bounding Box", resized_sample, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Bounding Box")
print('bounding box :', bounding_box)

x, y, w, h = [int(coord / (scale_percent / 100)) for coord in bounding_box]

image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not open image {input_path}.")
        continue

    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped and saved image: {output_path}")

print(f"Processed and cropped {len(image_files)} images using the bounding box.")