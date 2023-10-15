'''dahyun+darwin = dahwin'''
from rembg import remove
from PIL import Image

input_path = "dahyun.jpeg"
output_path = 'dahwin.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)