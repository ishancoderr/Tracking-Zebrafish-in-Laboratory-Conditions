import cv2
import os


input_videos = [
    ('./video/batch1_day2_1753_1805.mp4', (0, 0, 1753, 1805)),  # Coordinates for first video
    ('./video/batch3_day7_1829_1847.mp4', (0, 0, 1829, 1847))   # Coordinates for second video
]

output_folder = './cropped_frames/'  #output

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0


for input_video, (x, y, w, h) in input_videos:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}.")
        continue  

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        output_filename = f"{output_folder}cropped_frame_{frame_count}.jpg"
        cv2.imwrite(output_filename, cropped_frame)
        
        frame_count += 1
    cap.release()

cv2.destroyAllWindows()

print(f"Cropped {frame_count} frames and saved to {output_folder}.")