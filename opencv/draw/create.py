from PIL import Image

# Create a new black image with size 1920x1080
black_image = Image.new('RGB', (1920, 1080), color='black')

# Save the image
black_image.save('image.png')
from PIL import ImageColor

# Get all available color names
color_names = list(ImageColor.colormap.keys())

# Sort the color names alphabetically
color_names.sort()

# # Print all color names
# for color in color_names:
#     print(color)

# # Print the total number of colors
# print(f"\nTotal number of colors: {len(color_names)}")