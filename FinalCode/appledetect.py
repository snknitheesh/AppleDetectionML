from ultralytics import YOLO
import cv2, os
import numpy as np

model = YOLO("/home/zozo/workspaces/appledetection_ws/apple_ws/yolov5/yolov8s.pt")         
data_yaml = "/home/zozo/workspaces/appledetection_ws/apple_ws/final/apple.yaml" 
epochs = 100                    
batch_size = 32                     
img_size = 640                    

model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size)

val_results = model.val()
print(f"mAP@0.5: {val_results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

def apply_nms(boxes, scores, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
    return indices

test_image_folder = "/home/zozo/workspaces/appledetection_ws/apple_ws/yolov5/data/images"
results = model(test_image_folder, save=True, save_txt=True)

output_folder = "annotated_images"
os.makedirs(output_folder, exist_ok=True)

for i, result in enumerate(results):
    img = result.orig_img 
    bboxes = result.boxes.xyxy.cpu().numpy()  
    scores = result.boxes.conf.cpu().numpy() 
    classes = result.boxes.cls.cpu().numpy() 

    confidence_threshold = 0.5
    valid_indices = [idx for idx, score in enumerate(scores) if score >= confidence_threshold]
    bboxes = bboxes[valid_indices]
    scores = scores[valid_indices]
    classes = classes[valid_indices]

    selected_indices = apply_nms(bboxes.tolist(), scores.tolist())
    filtered_bboxes = [bboxes[idx] for idx in selected_indices]
    filtered_scores = [scores[idx] for idx in selected_indices]
    filtered_classes = [classes[idx] for idx in selected_indices]

    for bbox, score, cls in zip(filtered_bboxes, filtered_scores, filtered_classes):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"Class {int(cls)}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    num_bboxes = len(filtered_bboxes)
    print(f"Image {i + 1}: {num_bboxes} bounding boxes detected.")

    text = f"Apples detected: {num_bboxes}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = 10
    text_y = 30
    cv2.rectangle(img, (text_x - 5, text_y - 25), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)  # Background
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Text


    output_path = os.path.join(output_folder, f"image_{i + 1}.jpg")
    cv2.imwrite(output_path, img)

print(f"Annotated images saved in {output_folder}")

