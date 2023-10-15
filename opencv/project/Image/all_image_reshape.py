from PIL import Image
import os

# Define the new size for the images
new_size = (128, 128)

# Set the path to the folder containing the images
folder_path = r"C:\Users\Pc\Desktop\conputer_Vison\computervision_project\opencv\project\Image\image"

# Loop through each image in the folder
for filename in os.listdir(folder_path):
    # Open the image
    img = Image.open(os.path.join(folder_path, filename))

    # Resize the image
    img = img.resize(new_size)

    # Save the resized image
    img.save(os.path.join(folder_path, filename))
