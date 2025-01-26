from ultralytics import YOLO

# Parameters for the model
model = YOLO("yolov8s.pt")          # Pretrained Weights
data_yaml = "/home/zozo/workspaces/appledetection_ws/yolov5/apple.yaml"  # Path to your dataset YAML file
epochs = 200                         # Number of training epochs
batch_size = 32                     # Batch size
img_size = 640                      # Input image size for training

# Train the model
model.train(data=data_yaml, epochs=epochs, batch=batch_size, imgsz=img_size)

# Evaluate the model performance on validation data
val_results = model.val()
print(val_results)

# Test the model 
test_image = "/home/zozo/workspaces/appledetection_ws/yolov5/data/images"
results = model(test_image, save=True, save_txt=True)  
for result in results:
    result.show()               

# Save the annotated output
output_path = "runs/detect"
print(f"Results saved in {output_path}")

         
'''
Step1:
For - yolov5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

For - yolov8
pip install ultralytics
python3 -c "import ultralytics; print('ultralytics is installed!')" --verify installation
-----------------------------------------------------------------------------------------------------------------------------------------

Step2: Data collection

Data1:Source: https://www.kaggle.com/datasets/hossamfakher/apple-and-orange-yolo-detection/data
Data1:Notes: Take only apple data, labels and ignore orange data

Data2:Source: https://www.kaggle.com/datasets/projectlzp201910094/applebbch81?select=images
Data2:Notes: Take all the data and labels

Data3:Source: https://www.kaggle.com/datasets/snehamahata/apple-quality-dataset
Data3:Notes: Take all the data and labels

Data4: MinneApple:Source: https://conservancy.umn.edu/items/e1bb4015-e92a-4295-822c-d21d277ecfbd
Data4: MinneApple:Notes: Take the data inside detection.tar.gz. Use the segmentation images to create labels using bounding box
        Code for that is in convertMask.Py 

-----------------------------------------------------------------------------------------------------------------------------------------

Step3: Data formatting

Data Structure:
-Data
----train
--------images
--------labels
----valid
--------images
--------labels

----------------------------------------------------------------------------------------------------------------------------------------

Step4: Data rename(optional)
After adding all the images and label in the same folder, run  changeNames.py to rename it all to a songle class.
'''