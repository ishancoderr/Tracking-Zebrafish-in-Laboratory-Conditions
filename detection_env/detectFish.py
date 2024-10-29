import cv2
import numpy as np
import os

input_folder = './new_cropped_frames/'

image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])


image = cv2.imread(f"./new_cropped_frames/{image_files[1]}", cv2.IMREAD_GRAYSCALE)


blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

def update_edges(*args):
    min_thresh = cv2.getTrackbarPos('minThres', 'canny')
    max_thresh = cv2.getTrackbarPos('maxThres', 'canny')
    edges = cv2.Canny(blurred_image, min_thresh, max_thresh)
    cv2.imshow('canny', edges)

cv2.namedWindow('canny')

cv2.createTrackbar('minThres', 'canny', 0, 255, update_edges)
cv2.createTrackbar('maxThres', 'canny', 0, 255, update_edges)

update_edges()
cv2.waitKey(0)
cv2.destroyAllWindows()
