from pathlib import Path
import cv2
import numpy as np

def preprocess_to_binary(image, binary_thresh, background):
    '''
    Converts 2D image to binary after rescaling pixel intensity
    image: 2D np array
    binary_thresh: number from 0 - 1, multiplies the background before thresholding
    background: 2D background image (gleiche Maße wie image)
    '''
    image_rescale = image  # passt hier bereits
    min_difference = 5
    threshold = binary_thresh * background
    threshold = np.where(threshold < min_difference, min_difference, threshold)
    binary_image = np.where(image_rescale < threshold, 0, 255)
    return binary_image

img_path = Path("/Users/cara/Desktop/BA/YOLO/mask_to_yolo_test/GH039843_05015.jpg")
out_path = Path("/Users/cara/Desktop/BA/YOLO/mask_to_yolo_test/GH039843_05015_mask.png")

gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {img_path}")
gray = gray.astype(np.float32)

background = cv2.GaussianBlur(gray, (51, 51), 0)  # Approximation des Hintergrunds
binary_mask = preprocess_to_binary(gray, binary_thresh=0.7, background=background).astype(np.uint8)

cv2.imwrite(str(out_path), binary_mask)
print(f"Maske gespeichert unter {out_path}")
