import os
import shutil

# Set the path to the "twice" folder
source_folder = r"D:\twice\dawin_data"

# Set the path to the "image" folder
path_to_image = os.path.join(source_folder, "data")

# Create the "image" folder if it doesn't exist
if not os.path.exists(path_to_image):
    os.makedirs(path_to_image)

# Initialize the count of images to 0
image_count = 0

# Traverse through the "twice" folder and its subdirectories
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # Check if the file is an image file
        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
            # Get the file extension
            file_ext = os.path.splitext(file)[1]

            # Set the new filename for the image
            new_filename = str(image_count) + file_ext

            # Set the source and destination paths for the file
            src_path = os.path.join(root, file)
            dest_path = os.path.join(path_to_image, new_filename)

            # Copy the file to the "image" folder with the new filename
            shutil.copy(src_path, dest_path)

            # Increment the count of images
            image_count += 1
