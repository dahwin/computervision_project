'''dahyun+darwin = dahwin'''
import os
import glob

# replace "path/to/folder" with the path to your folder containing images
path = 'G:\\download\\Compressed\\train\\pizza'

# create a list of files in the folder with the extension ".jpg"
jpg_files = glob.glob(os.path.join(path, "*.jpg"))

# create a list of files in the folder with the extension ".png"
png_files = glob.glob(os.path.join(path, "*.png"))

# combine the two lists of files
image_files = jpg_files + png_files

# print the number of image files
print("Number of image files: ", len(image_files))
