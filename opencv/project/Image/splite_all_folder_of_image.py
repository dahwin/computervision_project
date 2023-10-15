import os
import shutil

# replace "D:\\dahyun\\Photobook" with the path to your folder containing images
# source_folder = "D:\\dahyun\\Photobook"
source_folder = "D:\\dahyun\\Photobook\\try"

# create a new folder to hold the training images
train_folder = os.path.join(source_folder, "80%folder")
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# create a new folder to hold the validation images
validation_folder = os.path.join(source_folder, "20%folder")
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# loop over all directories and files in the source folder
counter = 1
used_filenames = set()
for root, dirs, files in os.walk(source_folder):
    # filter the files list to only include image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for filename in image_files:
        # calculate the index to split the image files
        total_image_files = len(image_files)
        split_index = int(0.8 * total_image_files)

        # split the filename into its name and extension parts
        name, ext = os.path.splitext(filename)

        # construct the source and target paths
        source_path = os.path.join(root, filename)
        if counter >= split_index:
            target_name = f"{counter}{ext}"
            target_path = os.path.join(train_folder, target_name)
        else:
            target_name = f"{counter}{ext}"
            target_path = os.path.join(validation_folder, target_name)

        # check if the filename has already been used
        while target_name in used_filenames:
            counter += 1
            target_name = f"{counter}{ext}"
            if counter <= split_index:
                target_path = os.path.join(train_folder, target_name)
            else:
                target_path = os.path.join(validation_folder, target_name)

        # add the filename to the set of used filenames
        used_filenames.add(target_name)

        # copy the file to the appropriate folder
        shutil.copyfile(source_path, target_path)

        # increment the counter
        counter += 1


