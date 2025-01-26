import cv2
import numpy as np

def visualize_all_predictions(output_file, image_dir):
    # Initialize a dictionary to store all predictions for each image
    image_predictions = {}

    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Iterate through each line in the output file
    for line in lines:
        # Parse each line: image_name, xmin, ymin, xmax, ymax, score
        parts = line.strip().split(',')
        image_name = parts[0]
        xmin, ymin, xmax, ymax = map(int, parts[1:5])
        score = float(parts[5])

        # If the image name is not in the dictionary, add it
        if image_name not in image_predictions:
            image_predictions[image_name] = []

        # Append the prediction (bounding box and score) for the image
        image_predictions[image_name].append((xmin, ymin, xmax, ymax, score))

    # Now go through each image and visualize the predictions
    for image_name, predictions in image_predictions.items():
        # Load the image
        img_path = f"{image_dir}/{image_name}"
        image = cv2.imread(img_path)

        if image is not None:
            # Iterate through each prediction and draw the bounding box on the image
            for xmin, ymin, xmax, ymax, score in predictions:
                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green box

                # Add text with score
                cv2.putText(image, f"{score:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show image with bounding boxes
            cv2.imshow(f"Predictions for {image_name}", image)
            cv2.waitKey(0)  # Wait for key press to continue to the next image
            cv2.destroyAllWindows()
        else:
            print(f"Image {image_name} not found.")

output_file = "/home/zozo/workspaces/appledetection_ws/apple_ws/MinneApple/output/200E_frcnn_Minne_predictions/results.txt"  # Path to your output file
image_dir = "/home/zozo/workspaces/appledetection_ws/yolov5/data/images" 


visualize_all_predictions(output_file, image_dir)


