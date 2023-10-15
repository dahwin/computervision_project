'''dahyun+darwin = dahwin'''
import os
import shutil

# replace "path/to/source/folder" with the path to your folder containing images
source_folder = "D:\\dahyun\\Photobook\\1"

# create a new folder to hold the training images
train_folder = os.path.join(source_folder, "80%folder")
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# create a new folder to hold the validation images
validation_folder = os.path.join(source_folder, "20%folder")
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# get a list of all the image files with .png, .jpeg, and .jpg extensions in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))
               and f.lower().endswith(('.png', '.jpeg', '.jpg'))]

# calculate the index to split the image files
split_index = int(0.8 * len(image_files))

# copy the first 80% of image files to the train folder
for i in range(split_index):
    source_path = os.path.join(source_folder, image_files[i])
    target_path = os.path.join(train_folder, image_files[i])
    shutil.copyfile(source_path, target_path)

# copy the next 20% of image files to the validation folder
for i in range(split_index, len(image_files)):
    source_path = os.path.join(source_folder, image_files[i])
    target_path = os.path.join(validation_folder, image_files[i])
    shutil.copyfile(source_path, target_path)

