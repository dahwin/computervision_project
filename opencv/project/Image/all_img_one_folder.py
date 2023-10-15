import os
import shutil

# Set the source folder and destination folder
source_folder = "D:\\twice\\tuzyu"
# destination_folder = os.path.join(os.path.dirname(source_folder), "dahyun")
destination_folder = os.path.join(source_folder, "tuzyu")


# Create the destination folder if it doesn't already exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Initialize the file counter
file_counter = 1

# Loop through all the files in the source folder and its subfolders
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        # Get the file extension
        extension = os.path.splitext(filename)[1]

        # Check if the file has a valid extension
        if extension.lower() in ['.jpg', '.jpeg', '.png']:
            # Build the new filename with the file counter
            new_filename = str(file_counter) + extension

            # Build the full path of the source file and destination file
            source_file_path = os.path.join(root, filename)
            destination_file_path = os.path.join(destination_folder, new_filename)

            # Copy the file to the destination folder with the new filename
            shutil.copy(source_file_path, destination_file_path)

            # Increment the file counter
            file_counter += 1