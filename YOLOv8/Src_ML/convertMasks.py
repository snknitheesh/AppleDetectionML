import cv2
import os
import numpy as np

def mask_to_bbox(mask_path):
    # Read mask as grayscale image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Failed to load mask: {mask_path}")
        return []

    print("Mask shape:", mask.shape)

    # No thresholding needed since the mask is already in binary format

    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # Get bounding box coordinates from the contour
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Bounding box: {x}, {y}, {x + w}, {y + h}")
        bboxes.append([x, y, x + w, y + h])  # (xmin, ymin, xmax, ymax)
    
    return bboxes

def save_yolo_annotation(image_path, bboxes, output_path):
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            # Convert to YOLO format (normalized: center_x, center_y, width, height)
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape
            center_x = (bbox[0] + bbox[2]) / 2 / img_width
            center_y = (bbox[1] + bbox[3]) / 2 / img_height
            width = (bbox[2] - bbox[0]) / img_width
            height = (bbox[3] - bbox[1]) / img_height
            f.write(f"0 {center_x} {center_y} {width} {height}\n")  # Assuming class 0 for apple

def process_train_data(train_image_folder, mask_folder, annotation_folder):
    image_files = os.listdir(train_image_folder)
    for image_file in image_files:
        # Corresponding mask file
        mask_file = os.path.join(mask_folder, image_file)
        image_file_path = os.path.join(train_image_folder, image_file)
        
        # Convert mask to bounding boxes
        bboxes = mask_to_bbox(mask_file)
        
        # Save annotations in YOLO format
        annotation_file = os.path.join(annotation_folder, image_file.replace('.png', '.txt'))  # assuming .jpg
        save_yolo_annotation(image_file_path, bboxes, annotation_file)

train_image_folder = '/home/zozo/workspaces/appledetection_ws/MinneAppleData/detection/train/images'
mask_folder = '/home/zozo/workspaces/appledetection_ws/MinneAppleData/detection/train/masks'
annotation_folder = '/home/zozo/workspaces/appledetection_ws/MinneAppleData/detection/train/annotated_custom'

process_train_data(train_image_folder, mask_folder, annotation_folder)
