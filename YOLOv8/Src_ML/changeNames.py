import os

# Define the paths to your folders
images_folder = "/home/zozo/workspaces/appledetection_ws/apple-data (copy)/valid/images"
labels_folder = "/home/zozo/workspaces/appledetection_ws/apple-data (copy)/valid/labels"

# Get sorted lists of filenames from both folders
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))])
label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith(".txt")])

# Check if the number of files in both folders matches
if len(image_files) != len(label_files):
    print("Error: The number of image files and label files do not match!")
    exit()

# Rename images and labels
for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    # Check if filenames (without extensions) match
    if os.path.splitext(image_file)[0] != os.path.splitext(label_file)[0]:
        print(f"Error: Mismatched files: {image_file} and {label_file}")
        exit()

    # Define the new base name (e.g., image_001.jpg, image_001.txt)
    new_base_name = f"image_{i+1:03d}"  # Sequentially numbered with leading zeros
    new_image_name = new_base_name + ".jpg"
    new_label_name = new_base_name + ".txt"

    # Rename the files
    os.rename(os.path.join(images_folder, image_file), os.path.join(images_folder, new_image_name))
    os.rename(os.path.join(labels_folder, label_file), os.path.join(labels_folder, new_label_name))

    print(f"Renamed: {image_file} -> {new_image_name}, {label_file} -> {new_label_name}")

print("Renaming complete!")
