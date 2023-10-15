# import os
# import shutil
#
# # Set the path to the source folder
# source_folder = "path/to/source/folder"
#
# # Create the test folder and train folder
# test_folder = os.path.join(source_folder, "test")
# train_folder = os.path.join(source_folder, "train")
# os.makedirs(test_folder, exist_ok=True)
# os.makedirs(train_folder, exist_ok=True)
#
# # Get a list of all the image files with .png, .jpeg, and .jpg extensions in the source folder
# image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))
#                and f.lower().endswith(('.png', '.jpeg', '.jpg'))]
#
# # Calculate the index to split the image files
# split_index = int(0.2 * len(image_files))
#
# # Move the first 20% of image files to the test folder
# for i in range(split_index):
#     source_path = os.path.join(source_folder, image_files[i])
#     target_path = os.path.join(test_folder, image_files[i])
#     shutil.move(source_path, target_path)
#
# # Move the remaining 80% of image files to the train folder
# for i in range(split_index, len(image_files)):
#     source_path = os.path.join(source_folder, image_files[i])
#     target_path = os.path.join(train_folder, image_files[i])
#     shutil.move(source_path, target_path)
import os
import shutil

# Define the input and output folders
input_folder =  "D:\\twice\\vision"
output_folder = "D:\\twice\\z"

# Create the test and train folders
os.makedirs(os.path.join(output_folder, 'test'))
os.makedirs(os.path.join(output_folder, 'train'))

# Traverse the directory structure and copy the images to the appropriate folders
for dirpath, dirnames, filenames in os.walk(input_folder):
    for filename in filenames:
        if filename.endswith(('.jpeg', '.png', '.jpg')):
            # Get the folder name and create the output folder
            folder_name = os.path.basename(dirpath)
            output_folder_path = os.path.join(output_folder,
                                              'test' if len(filenames) * 0.2 >= filenames.index(filename) else 'train',
                                              folder_name)
            os.makedirs(output_folder_path, exist_ok=True)

            # Copy the image to the output folder
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(output_folder_path, filename)
            shutil.copyfile(src_path, dst_path)
