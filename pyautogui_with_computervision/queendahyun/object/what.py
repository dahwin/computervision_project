%%writefile dahwin_server.py
import paddle
from IPython.display import display
import os

# Get the current working directory
home = os.getcwd()


def print_available_gpus():
    print(f"PaddlePaddle version: {paddle.__version__}")
    
    devices = paddle.device.get_available_device()
    print("Available devices:", devices)
    print(f"Type of devices: {type(devices)}")
    
    if not devices:
        print("No devices detected!")
    
    for device in devices:
        print(f"- {device}")
    
    print(f"\nCUDA available: {paddle.device.is_compiled_with_cuda()}")
    
    if paddle.device.is_compiled_with_cuda():
        gpu_count = paddle.device.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        for i in range(gpu_count):
            try:
                gpu_name = paddle.device.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
            except Exception as e:
                print(f"  Error getting name for GPU {i}: {str(e)}")

if __name__ == "__main__":
    print_available_gpus()

os.chdir(f'{home}/exe')
from cryptography.fernet import Fernet
import tempfile
import os
import paddle

from paddleoc import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import shutil

# Directories containing encrypted model files
secure_models_dir = f"{home}/secure_models"



def decrypt_model_files(encrypted_model_path, temp_model_path, key):
    cipher_suite = Fernet(key)
    file_mapping = {
        "0.dubu": "inference.pdiparams",
        "1.dubu": "inference.pdiparams.info",
        "2.dubu": "inference.pdmodel"
    }

    for encrypted_file_name, decrypted_file_name in file_mapping.items():
        with open(os.path.join(encrypted_model_path, encrypted_file_name), "rb") as file:
            encrypted_model_data = file.read()
        decrypted_model_data = cipher_suite.decrypt(encrypted_model_data)
        with open(os.path.join(temp_model_path, decrypted_file_name), "wb") as file:
            file.write(decrypted_model_data)

# Create temporary directories and decrypt model files
temp_dirs = {}
for model_name in os.listdir(secure_models_dir):
    print(model_name)
    encrypted_model_path = os.path.join(secure_models_dir, model_name)
    temp_model_path = tempfile.mkdtemp()
    temp_dirs[model_name] = temp_model_path
    # key = load_encryption_key(encrypted_model_path)
    if model_name=="progress0":
        key = "XZ3Pjc_HPVcgmWpaAKGAVHiHg6zXwgJxKRuG6yAgGpg="
    if model_name=="progress1":
        key = "FcwIsmP8gdcFY1BeDm6ogPHKGhgKrjcRR3hW-AjwjvI="
    if model_name =="progress2":
        key = "IkpQYP9x_1dmcSkafNFn13tl6-43rkY9AJAZsTmj_xE="
    decrypt_model_files(encrypted_model_path, temp_model_path, key)

# Load the PaddleOCR models
mocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, precision='fp16', gpu_id=0,
                det_model_dir=temp_dirs['progress0'],
                rec_model_dir=temp_dirs['progress1'],
                cls_model_dir=temp_dirs['progress2'])
# Clean up temporary directories
for temp_dir in temp_dirs.values():
    shutil.rmtree(temp_dir)

# Perform OCR on the image
img_path = '/kaggle/input/ddddddddddd/agi-.png'
img_path =f"{home}/keyl.png"
result = mocr.ocr(img_path, cls=True)

# Load the image using Pillow
image = Image.open(img_path)
draw = ImageDraw.Draw(image)
bresult = []

# Iterate through the OCR results
for res in result:
    for line in res:
        box, text, conf = line[0], line[1][0], line[1][1]
        box = [(int(pt[0]), int(pt[1])) for pt in box]
        x_coordinates = [point[0] for point in box]
        y_coordinates = [point[1] for point in box]
        x1 = min(x_coordinates)
        y1 = min(y_coordinates)
        x2 = max(x_coordinates)
        y2 = max(y_coordinates)
        bresult.append([x1, y1, x2, y2, text, conf])
        draw.polygon(box, outline=(0, 255, 0))
        font = ImageFont.load_default()
        draw.text((box[0][0], box[0][1]), text, fill=(0, 0, 255), font=font)




# Display the image with OCR results
display(image)


import time
def dahwin_ocr(img_list):
    global alo
    global answer_d
    
    
    s = time.time()
    bounding_boxes = []
    for img in img_list:
        al = []
        original_image = img
        # Get the dimensions of the original image
        height, width, _ = original_image.shape

        # Calculate the midpoint along the width
        midpoint = width // 2

        # Split the image into left and right halves
        left_half = original_image[:, :midpoint]
        right_half = original_image[:, midpoint:]

        result = mocr.ocr(left_half, cls=True)
        bresult = []
        bresult1 = []
        txt = []
        txt1 = []
        b = []
        b1 = []

        # Iterate through the OCR results
        for res in result:
            for line in res:
                box, text,conf = line[0], line[1][0],line[1][1]
                # Convert box coordinates to integers
                box = [(int(pt[0]), int(pt[1])) for pt in box]

                x_coordinates = [point[0] for point in box]
                y_coordinates = [point[1] for point in box]
                x1 = min(x_coordinates)
                y1 = min(y_coordinates)
                x2 = max(x_coordinates)
                y2 = max(y_coordinates)
#                 bresult.append([x1, y1, x2, y2,text,conf])
                txt.append(text)
                b.append([x1, y1, x2, y2])

#                 print(text)
                al.append((box, text))  # Append bounding box coordinates and class label
        bounding_boxes.append(al)
        al.clear()
        
        
        
        result = mocr.ocr(right_half, cls=True)
        

        # Iterate through the OCR results
        for res in result:
            for line in res:
                box, text,conf = line[0], line[1][0],line[1][1]
                # Convert box coordinates to integers
                box = [(int(pt[0]), int(pt[1])) for pt in box]

                x_coordinates = [point[0] for point in box]
                y_coordinates = [point[1] for point in box]
                x1 = min(x_coordinates)
                y1 = min(y_coordinates)
                x2 = max(x_coordinates)
                y2 = max(y_coordinates)
                bresult1.append([x1, y1, x2, y2,text,conf])
                al.append((box, text))  # Append bounding box coordinates and class label
        bounding_boxes.append(al)
        al.clear()
        
        
        
        
        original_width = 1920


        for x1,y1,x2,y2,text,conf in bresult1:

            x1,y1,x2,y2 = [x1 + original_width // 2,y1,
                x2 + original_width // 2,y2
            ]
            txt.append(text)
            b.append([x1, y1, x2, y2])

        
    answer_d=[b,txt]
    

    
    txt.append(text)
        
    print(f"result {len(bounding_boxes)}")
    
    print(f'bresult {len(bresult)}')
    e = time.time()
    l = e-s
    print(l)
    return answer_d





import cv2
img_path = f"{home}/subscribe.png"
img_path = f"{home}/word.png"
img = cv2.imread(img_path)
img_list = [img]
ocr_b,ocr = dahwin_ocr(img_list)
print(ocr)
print(ocr_b)




from PIL import Image, ImageDraw, ImageFont
import random

def draw_bbox_and_label(draw, bbox, label, color):
    x1, y1, x2, y2 = map(int, bbox)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    # Use default font
    font = ImageFont.load_default()
    
    # Get label size using textbbox
    label_bbox = draw.textbbox((x1, y1), label, font=font)
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]
    
    # Draw filled rectangle for label background
    draw.rectangle([x1, y1 - label_height - 5, x1 + label_width, y1], fill=color)
    
    # Put text on the filled rectangle
    draw.text((x1, y1 - label_height - 5), label, fill=(255, 255, 255), font=font)

# Load and resize the image
image_path = img_path 
image = Image.open(image_path)
image = image.resize((1920, 1080))

# Create a draw object
draw = ImageDraw.Draw(image)

# Get data for top_left_corner
objects = ocr  # Placeholder for OCR-detected objects
bboxes = ocr_b  # Placeholder for bounding boxes of objects

# Draw bounding boxes and labels for top_left_corner
for obj, bbox in zip(objects, bboxes):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw_bbox_and_label(draw, bbox, obj, color)

# Save the image
output_path = f"{home}/output_image_with_bbox_top_left.png"
image.save(output_path)
print(f"Image saved to {output_path}")

# Display the image (this will work in a Jupyter notebook)
display(image)












os.chdir(home)


import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ram.ram import RAM as ram
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
image_size=384
# Load RAM model weights in FP16 precision
def load_ram_fp16(model, load_path, device):
    # Load the state dict
    state_dict = torch.load(load_path, map_location=device)
    
    # Convert the model to fp16
    model.half()
    
    # Load the state dict
    model.load_state_dict(state_dict)
    
    return model


def inference(image, model):

    with torch.no_grad():
        tags, tags_chinese = model.generate_tag(image)

    return tags[0],None
# Example usage to load the model
def load_ram_model_fp16(pretrained_path, image_size=384, vit='swin_l', device='cuda'):
    model = ram(image_size=image_size, vit=vit)
    model = load_ram_fp16(model, pretrained_path, device)
    model.eval()
    model = model.to(device)
    return model

# Usage example
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device='cuda:0'
fp16_model_path = f"{home}/ram_swin_large_14m_fp16.pth'
model_fp16 = load_ram_model_fp16(fp16_model_path, device=device)
import shutil

shutil.rmtree(f"{home}/ram/c/sm")
# Inference with FP16 model
def inference_fp16(img, model, image_size=384, device='cuda:0'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    if isinstance(img, Image.Image):
        raw_image = img.convert("RGB").resize((image_size, image_size))
    else:
        raw_image = Image.open(img).convert("RGB").resize((image_size, image_size))
        
    
    
    image = transform(raw_image).unsqueeze(0).to(device).half()  # Convert to FP16
    
    with torch.no_grad():
        res = inference(image, model)
    
    return res[0]



# Example usage
image_path = f"{home}/Picsart_24-03-16_16-25-10-385 (1) (1).jpg"
tags = inference_fp16(image_path, model_fp16)
print("Image Tags:", tags)
































from PIL import Image
img = Image.open(f"{home}/twice.png").resize((1280,720))
print(img.size)
img.save(f"{home}/twice.png")


import torch
from PIL import Image
import io
import os
import supervision as sv
import numpy as np
import requests
import cv2

# Grounding DINO
from groundingdino.util.inference import BatchedModel
import torchvision.transforms.functional as F
from huggingface_hub import hf_hub_download

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

def load_image(image_path):
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_tensor = F.to_tensor(image)
    return image, image_tensor

local_image_path = f"{home}/twice.png"
#download_image(image_url, local_image_path)
image_source, image_tensor = load_image(local_image_path)
display(Image.fromarray(image_source))

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swint_ogc.pth"
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
device = "cuda:0"

cache_config_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename)
cache_file = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filename)

batch = 4

dtype = "float16"

# Building GroundingDINO inference model
grounding_dino_model = BatchedModel(
    model_config_path=cache_config_file,
    model_checkpoint_path=cache_file,
    device=device,
    dtype=dtype
)
# Image paths
image_paths = [
    f"{home}/twice.png",
#     "/kaggle/working/80673f7f0fe96ecf05c69abe27cb05b9.1000x1000x1.png",
#  "/kaggle/working/80673f7f0fe96ecf05c69abe27cb05b9.1000x1000x1.png",
# "/kaggle/working/80673f7f0fe96ecf05c69abe27cb05b9.1000x1000x1.png",
]



# text_prompt = [
#     ["flower",'girl']]*len(image_paths)

text_prompt = []
import re
for idx,img_path in enumerate(image_paths):
    tags = inference_fp16(img_path, model_fp16)
    # Use re.findall to extract all words, including multi-word tags
    tag_list = re.findall(r'\w+(?:\s+\w+)*', tags)
    text_prompt.append(tag_list)
  
print(text_prompt)

# Batch processing
images_tensors = []
for image_path in image_paths:
    _, image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(device=device).to(dtype=getattr(torch, dtype))
    images_tensors.append(image_tensor)

image_tensor_batch = torch.stack(images_tensors)
box_threshold = 0.07
text_threshold =0.07
iou_threshold = 0.5
import time
s = time.time()
with torch.no_grad():
    bbox_batch, conf_batch, class_id_batch  = grounding_dino_model(
        image_batch=image_tensor_batch,
        text_prompts=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=iou_threshold
    )
#     flattened_bbox_batch = [box.flatten().tolist() for bbox in bbox_batch for box in bbox.cpu().numpy()]

#     flattened_conf_batch = [box.flatten().tolist() for bbox in conf_batch for box in bbox.cpu().numpy()]
    flattened_bbox_batch = [bbox.tolist() for bbox in bbox_batch]
    class_id_batch = [class_ids.tolist() for class_ids in class_id_batch]
    class_id = [[t[c] for c in cls] for t, cls in zip(text_prompt, class_id_batch)]



    
e=time.time()
l = e-s
print(l)
object_ = class_id[0]
object_b = flattened_bbox_batch[0]
# print(class_id_batch)
print(object_)
print(object_b)
print(len(class_id[0]))
print(len(flattened_bbox_batch[0]))
# print(class_id)
# print(flattened_bbox_batch)




from PIL import Image, ImageDraw, ImageFont
import random

# Load your image (replace with your actual path)
image_path = f"{home}/twice.png"
image = Image.open(image_path)

# Prepare to draw
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Use default font; you can specify a custom one


# Extract the first batch of class IDs and bounding boxes
class_id = class_id[0]
flattened_bbox_batch = flattened_bbox_batch[0]

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Iterate and draw bounding boxes with labels
for i in range(len(class_id)):
    # Get class and box coordinates
    class_name = class_id[i]
    x0, y0, x1, y1 = flattened_bbox_batch[i]

    # Generate a random color for each bounding box
    color = random_color()

    # Draw bounding box
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)

    # Draw label
    text_position = (x0, y0 - 10)  # Adjust position to be above the box
    draw.text(text_position, class_name, fill=color, font=font)

# Save or show the result
display(image)
# image.save("output.png")  # Save the result if needed





def load_image(image):
    image = np.asarray(image)
    image_tensor = F.to_tensor(image)
    return image, image_tensor

def dahwin_object(image_list):


    text_prompt = []
    import re
    for idx,img in enumerate(image_list):
        tags = inference_fp16(img, model_fp16)
        # Use re.findall to extract all words, including multi-word tags
        tag_list = re.findall(r'\w+(?:\s+\w+)*', tags)
        text_prompt.append(tag_list)

    print(text_prompt)

    # Batch processing
    images_tensors = []
    for image_path in image_list:
        _, image_tensor = load_image(image_path)
        image_tensor = image_tensor.to(device=device).to(dtype=getattr(torch, dtype))
        images_tensors.append(image_tensor)

    image_tensor_batch = torch.stack(images_tensors)

    box_threshold = 0.07
    text_threshold =0.07
    iou_threshold = 0.5
    import time
    s = time.time()
    with torch.no_grad():
        bbox_batch, conf_batch, class_id_batch  = grounding_dino_model(
            image_batch=image_tensor_batch,
            text_prompts=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=iou_threshold
        )
    #     flattened_bbox_batch = [box.flatten().tolist() for bbox in bbox_batch for box in bbox.cpu().numpy()]

    #     flattened_conf_batch = [box.flatten().tolist() for bbox in conf_batch for box in bbox.cpu().numpy()]
        flattened_bbox_batch = [bbox.tolist() for bbox in bbox_batch]
        class_id_batch = [class_ids.tolist() for class_ids in class_id_batch]
        class_id = [[t[c] for c in cls] for t, cls in zip(text_prompt, class_id_batch)]




    e=time.time()
    l = e-s
    print(l)
    object_ = class_id[0]
    object_b = flattened_bbox_batch[0]
    
    answer_d = [object_,object_b]
    return answer_d



from PIL import Image
image_path = f"{home}/twice.png"
img = Image.open(image_path).convert("RGB").resize((1280, 720))
object_,object_b = dahwin_object([img])
print(object_)
print(object_b)



def scale_bboxes(bboxes, original_size, target_size):
    """
    Scales bounding boxes from the original image size to the target image size.
    
    Parameters:
    - bboxes: List of bounding boxes, where each box is in the format [x0, y0, x1, y1].
    - original_size: Tuple of the original image size (width, height).
    - target_size: Tuple of the target image size (width, height).
    
    Returns:
    - List of scaled bounding boxes in the format [x0, y0, x1, y1].
    """
    original_width, original_height = original_size
    target_width, target_height = target_size
    
    # Calculate scale factors
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # Scale each bounding box
    scaled_bboxes = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        scaled_bbox = [
            x0 * scale_x,
            y0 * scale_y,
            x1 * scale_x,
            y1 * scale_y
        ]
        scaled_bboxes.append(scaled_bbox)
    
    return scaled_bboxes
original_size = (1280, 720)
target_size = (1920, 1080)

scaled_bboxes = scale_bboxes(object_b, original_size, target_size)
print(scaled_bboxes)




from PIL import Image, ImageDraw, ImageFont
import random


image = Image.open(image_path).resize((1920,1080))

# Prepare to draw
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Use default font; you can specify a custom one


# Extract the first batch of class IDs and bounding boxes
class_id = object_
flattened_bbox_batch = scale_bboxes(object_b, original_size, target_size)

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Iterate and draw bounding boxes with labels
for i in range(len(class_id)):
    # Get class and box coordinates
    class_name = class_id[i]
    x0, y0, x1, y1 = flattened_bbox_batch[i]

    # Generate a random color for each bounding box
    color = random_color()

    # Draw bounding box
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)

    # Draw label
    text_position = (x0, y0 - 10)  # Adjust position to be above the box
    draw.text(text_position, class_name, fill=color, font=font)

# Save or show the result
display(image)
# image.save("output.png")  # Save the result if needed








































from cryptography.fernet import Fernet
import tempfile
import os
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont

# Custom function to load and decrypt the model
def load_encrypted_model(model_path, key):
    cipher_suite = Fernet(key)

    # Read the encrypted model file
    with open(model_path, "rb") as file:
        encrypted_model_data = file.read()

    # Decrypt the model data
    decrypted_model_data = cipher_suite.decrypt(encrypted_model_data)

    # Temporarily save the decrypted model to a file with correct suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        temp_file.write(decrypted_model_data)
        temp_model_path = temp_file.name

    # Load the model using Ultralytics
    model = YOLO(temp_model_path)

    # Optionally delete the temporary file after loading
    os.remove(temp_model_path)

    return model

# Load the encryption key
key = "bMNT5nrrMdrxBnaXTwD4oHVpBSRyrSQ8dP1FfJNfiB8="

# Load the encrypted model
modeld = load_encrypted_model("consolidated1.pth", key)
modeld = modeld.to('cuda:1')
# Path to the image
image_path = f"{home}/keyl.png"

# Read the image using OpenCV
img = cv2.imread(image_path)

# Perform object detection
results = modeld.predict(img)

# List to store bounding boxes
bounding_boxes = []

# Iterate over results
for r in results:
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
        c = box.cls       # Get class label
        bounding_boxes.append((b, modeld.names[int(c)]))  # Append bounding box coordinates and class label

# Display the image with bounding boxes using Pillow
pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)
font = ImageFont.load_default()

# Draw bounding boxes on the image
for bbox, class_label in bounding_boxes:
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")
    draw.text((bbox[0], bbox[1]), class_label, fill="green", font=font)

# Display the image
display(pil_image)













from cryptography.fernet import Fernet
import tempfile
import os
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont

# Custom function to load and decrypt the model
def load_encrypted_model(model_path, key):
    cipher_suite = Fernet(key)

    # Read the encrypted model file
    with open(model_path, "rb") as file:
        encrypted_model_data = file.read()

    # Decrypt the model data
    decrypted_model_data = cipher_suite.decrypt(encrypted_model_data)

    # Temporarily save the decrypted model to a file with correct suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
        temp_file.write(decrypted_model_data)
        temp_model_path = temp_file.name

    # Load the model using Ultralytics
    model = YOLO(temp_model_path)

    # Optionally delete the temporary file after loading
    os.remove(temp_model_path)

    return model

# Load the encryption key
key = "bMNT5nrrMdrxBnaXTwD4oHVpBSRyrSQ8dP1FfJNfiB8="

# Load the encrypted model
modeld2 = load_encrypted_model("consolidated1.pth", key)
modeld2 = modeld2.to('cuda:0')





from PIL import Image, ImageDraw




def two_step_adjust(bbox, extracted_bbox):


    # Calculate the offset
    offset_x = extracted_bbox[0]
    offset_y = extracted_bbox[1]

    # Adjust the bbox coordinates
    adjusted_bbox = (
        bbox[0] + offset_x,
        bbox[1] + offset_y,
        bbox[2] + offset_x,
        bbox[3] + offset_y
    )


    return adjusted_bbox



def dahwin_icon(img_list):
    global answer_d
    answer_d =[]
#     bounding_boxes = []
    bounding_boxes = []
    clss = []
    s = time.time()
    for img in img_list:
        # Perform object detection
        results = modeld.predict(img)  # Wait for the prediction asynchronously

        # Iterate over results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
                b = b.tolist()  # Convert tensor to list
                c = box.cls       # Get class label
#                 bounding_boxes.append((b, modeld.names[int(c)]))  # Append bounding box coordinates and class label
                bounding_boxes.append((b))  # Append bounding box coordinates and class label
                clss.append(modeld.names[int(c)])
    answer_d=[bounding_boxes,clss]
    e = time.time()
    r = e-s
    print(f"total time {r}")
    print(f"result {len(bounding_boxes)}")
    return answer_d


















def dahwin_icon(img_list):
    img = img_list[0]
    image_p_list = []
    
    width, height = img.size
    mid_width = width // 2
    mid_height = height // 2

    # Define the box coordinates for the four pieces
    top_left_box = (0, 0, mid_width, mid_height)
    top_right_box = (mid_width, 0, width, mid_height)
    bottom_left_box = (0, mid_height, mid_width, height)
    bottom_right_box = (mid_width, mid_height, width, height)

    image_p_list.append(top_left_box)
    image_p_list.append(top_right_box)
    image_p_list.append(bottom_left_box)
    image_p_list.append(bottom_right_box)

    # Crop the image into four pieces
    top_left_piece = img.crop(top_left_box)
    top_right_piece = img.crop(top_right_box)
    bottom_left_piece = img.crop(bottom_left_box)
    bottom_right_piece = img.crop(bottom_right_box)
    
    
    imgl = [top_left_piece,top_right_piece,bottom_left_piece,bottom_right_piece]
            

    
    s = time.time()
    # Perform object detection
    results = modeld.predict(imgl)

    # List to store bounding boxes
    bounding_boxes = []
    clss = []
    # Iterate over results
    for idx,r in enumerate(results):
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()  # Get box coordinates and convert to list
#             print("Original bbox:", b)

            out = two_step_adjust(b, image_p_list[idx])
#             print("Adjusted bbox:", out)
            c = box.cls       # Get class label
            bounding_boxes.append((out))  # Append bounding box coordinates and class label
            clss.append( modeld.names[int(c)])
    e = time.time()
    r = e-s
    print(f"total time {r}")


    return [bounding_boxes,clss]
    

from PIL import Image
# Path to the image
image_path = f"{home}/TWICE_Image.jpg"
image_path = "/kaggle/input/folder/folder-.png"
image_path ="/kaggle/input/ddddddddddd/grid.png"
image_path = f"{home}/keyl.png"
image_path = f"{home}/p.png"
# Read the image using OpenCV
# img = cv2.imread(image_path)
imgg = Image.open(image_path)
# img = imgg.resize((1280, 720))
original_size =imgg.size
bounding_boxes,clss = dahwin_icon([imgg])
bounding_boxes = scale_bboxes(bounding_boxes, original_size, target_size)
pil_image =imgg
draw = ImageDraw.Draw(pil_image)
font = ImageFont.load_default()

# Draw bounding boxes on the image
for bbox, class_label in zip(bounding_boxes,clss):
# for bbox, class_label in bounding_boxes:
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")
    draw.text((bbox[0], bbox[1]), class_label, fill="green", font=font)

# Display the image
display(pil_image)







from PIL import Image, ImageDraw




def two_step_adjust(bbox, extracted_bbox):


    # Calculate the offset
    offset_x = extracted_bbox[0]
    offset_y = extracted_bbox[1]

    # Adjust the bbox coordinates
    adjusted_bbox = (
        bbox[0] + offset_x,
        bbox[1] + offset_y,
        bbox[2] + offset_x,
        bbox[3] + offset_y
    )


    return adjusted_bbox



def dahwin_icon(img_list):
    global answer_d
    answer_d =[]
#     bounding_boxes = []
    bounding_boxes = []
    clss = []
    s = time.time()
    for img in img_list:
        # Perform object detection
        results = modeld.predict(img)  # Wait for the prediction asynchronously

        # Iterate over results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
                b = b.tolist()  # Convert tensor to list
                c = box.cls       # Get class label
#                 bounding_boxes.append((b, modeld.names[int(c)]))  # Append bounding box coordinates and class label
                bounding_boxes.append((b))  # Append bounding box coordinates and class label
                clss.append(modeld.names[int(c)])
    answer_d=[bounding_boxes,clss]
    e = time.time()
    r = e-s
    print(f"total time {r}")
    print(f"result {len(bounding_boxes)}")
    return answer_d






def dahwin_icon(img_list):
    img = img_list[0]
            


    def first():
        img = img_list[0]
        image_p_list = []

        width, height = img.size
        mid_width = width // 2
        mid_height = height // 2

        # Define the box coordinates for the four pieces
        top_left_box = (0, 0, mid_width, mid_height)
        top_right_box = (mid_width, 0, width, mid_height)
        bottom_left_box = (0, mid_height, mid_width, height)
        bottom_right_box = (mid_width, mid_height, width, height)

        image_p_list.append(top_left_box)
        image_p_list.append(top_right_box)
        image_p_list.append(bottom_left_box)
        image_p_list.append(bottom_right_box)

        # Crop the image into four pieces
        top_left_piece = img.crop(top_left_box)
        top_right_piece = img.crop(top_right_box)
        bottom_left_piece = img.crop(bottom_left_box)
        bottom_right_piece = img.crop(bottom_right_box)


        imgl = [top_left_piece,top_right_piece,bottom_left_piece,bottom_right_piece]



        s = time.time()
        # Perform object detection
        results = modeld.predict(imgl)

        # List to store bounding boxes
        bounding_boxes = []

        # Iterate over results
        for idx,r in enumerate(results):
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()  # Get box coordinates and convert to list
    #             print("Original bbox:", b)

                out = two_step_adjust(b, image_p_list[idx])
    #             print("Adjusted bbox:", out)
                c = box.cls       # Get class label
                bounding_boxes.append((out, modeld.names[int(c)]))  # Append bounding box coordinates and class label
        e = time.time()
        r = e-s
        print(f"total time {r}")


        return bounding_boxes
    
    def second():
        start = time.time()        

        # Perform object detection
        results = modeld2.predict(img)

        # List to store bounding boxes
        bounding_boxes_1 = []

        # Iterate over results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
                c = box.cls       # Get class label
                # Convert tensor to list of floats and round to 2 decimal places
                bbox = [round(float(coord), 2) for coord in b.tolist()]
                bounding_boxes_1.append((bbox, modeld.names[int(c)]))  # Append bounding box coordinates and class label
        e = time.time()
        r = e-start
        print(f"bounding_boxes_1 total time {r}")
        return bounding_boxes_1
    
    mstart = time.time()
    bounding_boxes = first()
    bounding_boxes_1=second()
    # Run `first` and `second` functions concurrently
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(first), executor.submit(second)]
#         results = [future.result() for future in as_completed(futures)]
    
#     bounding_boxes =  results[0]
#     bounding_boxes_1 = results[1]
    
    
    all_box = bounding_boxes_1+bounding_boxes
    print(f"bounding_boxes_1 {len(bounding_boxes_1)}")
    
    
    print(f"bounding_boxes {len(bounding_boxes)}")
    
    
    print(f"total bboxes = {len(all_box)}")
    

    
    

    def remove_overlapped_boxes(all_box):
        """
        Remove overlapped bounding boxes from the list while keeping the first occurrence.

        Args:
            all_box: List of tuples, each containing (bbox, label) where bbox is [x1,y1,x2,y2]

        Returns:
            List of (bbox, label) tuples with overlapped boxes removed
        """
        filtered_boxes = []

        for i in range(len(all_box)):
            bbox1, label1 = all_box[i]
            overlap = False

            # Compare with previously accepted boxes
            for prev_bbox, prev_label in filtered_boxes:
                # Check for overlap
                if (bbox1[0] < prev_bbox[2] and bbox1[2] > prev_bbox[0] and
                    bbox1[1] < prev_bbox[3] and bbox1[3] > prev_bbox[1]):
                    overlap = True
                    break

            # If no overlap found, add to filtered boxes
            if not overlap:
                filtered_boxes.append((bbox1, label1))

        return filtered_boxes
    # Replace the overlapped boxes section with this:
    filtered_all_box = remove_overlapped_boxes(all_box)
    print(f"boxes after removing overlaps = {len(filtered_all_box)}")
    
    
    
    
    
    


    # List to store bounding boxes
    bounding_boxes = []
    clss = []
    # Draw bounding boxes on the image
    for bbox, class_label in filtered_all_box:
        clss.append(class_label)
        bounding_boxes.append(bbox)
      

    
    
    
    
    
    
    
    
    e = time.time()
    r = e-mstart
    print(f"total time {r}")
    


    return [bounding_boxes,clss]
    



    
    

    
    
    
    
    
    
    
    
    
    

    
image_path = f"{home}/p.png"


imgg = Image.open(image_path)

original_size =imgg.size
bounding_boxes,clss = dahwin_icon([imgg])
bounding_boxes = scale_bboxes(bounding_boxes, original_size, target_size)
pil_image =imgg
draw = ImageDraw.Draw(pil_image)
font = ImageFont.load_default()

# Draw bounding boxes on the image
for bbox, class_label in zip(bounding_boxes,clss):
# for bbox, class_label in bounding_boxes:
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")
    draw.text((bbox[0], bbox[1]), class_label, fill="green", font=font)

# Display the image
display(pil_image)









from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from PIL import Image, ImageDraw, ImageFont

def two_step_adjust(bbox, extracted_bbox):
    offset_x = extracted_bbox[0]
    offset_y = extracted_bbox[1]
    adjusted_bbox = (
        bbox[0] + offset_x,
        bbox[1] + offset_y,
        bbox[2] + offset_x,
        bbox[3] + offset_y
    )
    return adjusted_bbox

def dahwin_icon(img_list):
    img = img_list[0]
    image_p_list = []
    width, height = img.size
    mid_width = width // 2
    mid_height = height // 2

    # Define the box coordinates for the four pieces
    image_p_list = [
        (0, 0, mid_width, mid_height),
        (mid_width, 0, width, mid_height),
        (0, mid_height, mid_width, height),
        (mid_width, mid_height, width, height),
    ]

    # Split the image into four pieces
    imgl = [img.crop(box) for box in image_p_list]

    def first():
        s = time.time()
        results = modeld.predict(imgl)
        bounding_boxes = []

        for idx, r in enumerate(results):
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                out = two_step_adjust(b, image_p_list[idx])
                c = box.cls
                bounding_boxes.append((out, modeld.names[int(c)]))
        print(f"First detection time: {time.time() - s}")
        return bounding_boxes

    def second():
        start = time.time()
        results = modeld2.predict(img)
        bounding_boxes_1 = []

        for r in results:
            for box in r.boxes:
                b = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]
                c = box.cls
                bounding_boxes_1.append((b, modeld.names[int(c)]))
        print(f"Second detection time: {time.time() - start}")
        return bounding_boxes_1

    # Run `first` and `second` functions concurrently
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(first), executor.submit(second)]
        results = [future.result() for future in as_completed(futures)]

    bounding_boxes = results[0] + results[1]

    def remove_overlapped_boxes(all_box):
        filtered_boxes = []
        for i, (bbox1, label1) in enumerate(all_box):
            overlap = False
            for prev_bbox, _ in filtered_boxes:
                if (bbox1[0] < prev_bbox[2] and bbox1[2] > prev_bbox[0] and
                    bbox1[1] < prev_bbox[3] and bbox1[3] > prev_bbox[1]):
                    overlap = True
                    break
            if not overlap:
                filtered_boxes.append((bbox1, label1))
        return filtered_boxes

    filtered_all_box = remove_overlapped_boxes(bounding_boxes)
    print(f"Total boxes after overlap removal: {len(filtered_all_box)}")

    return [box for box, _ in filtered_all_box], [label for _, label in filtered_all_box]

# Example usage
image_path = f"{home}/p.png"
image_path = f"{home}/word.png"
imgg = Image.open(image_path)
original_size = imgg.size
bounding_boxes, clss = dahwin_icon([imgg])

# Draw bounding boxes on the image
draw = ImageDraw.Draw(imgg)
font = ImageFont.load_default()

for bbox, class_label in zip(bounding_boxes, clss):
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red")
    draw.text((bbox[0], bbox[1]), class_label, fill="green", font=font)

# Display the image
display(imgg)










icon = clss
print(icon)
icon_b = bounding_boxes
print(icon_b)











import cv2
from PIL import Image
import time
import threading

def process_ocr(img_list, ocr_result):
    ocr_b, ocr = dahwin_ocr(img_list)
    ocr_result.extend([ocr_b, ocr])

def process_icon(img, icon_result):
    bounding_boxes, clss = dahwin_icon([img])
    icon_result.extend([bounding_boxes, clss])

def process_object(img, object_result):
    object_, object_b = dahwin_object([img])
    object_result.extend([object_, object_b])

s = time.time()
main_path = f"{home}/subscribe.png"
main_path = "pixtral.png"
img = cv2.imread(main_path)
img_resized = cv2.resize(img, (1920, 1080))
img_list = [img_resized]

# Prepare PIL image
pil_img = Image.open(main_path)

pil_img_rgb = pil_img.convert("RGB").resize((1280, 720))

# Prepare result lists
ocr_result = []
icon_result = []
object_result = []

# Create threads
ocr_thread = threading.Thread(target=process_ocr, args=(img_list, ocr_result))
icon_thread = threading.Thread(target=process_icon, args=(pil_img, icon_result))
object_thread = threading.Thread(target=process_object, args=(pil_img_rgb, object_result))

# Start threads
ocr_thread.start()
icon_thread.start()
object_thread.start()

# Wait for all threads to complete
ocr_thread.join()
icon_thread.join()
object_thread.join()

# Unpack results
ocr_b, ocr = ocr_result
bounding_boxes, icon = icon_result
object_, object_b = object_result

print(ocr)
print(ocr_b)
print(icon)

original_size = pil_img.size
# Assuming you have original_size and target_size defined
icon_b = scale_bboxes(bounding_boxes, original_size, target_size)
print(icon_b)

original_size =pil_img_rgb.size

object_b = scale_bboxes(object_b, original_size, target_size)
print(object_)
print(object_b)

e = time.time()
r = e - s
print(f"total execution time: {r}")













pil_img_rgb.size,pil_img.size
print(len(ocr),len(ocr_b))

# Example usage
image_path = f"{home}/Picsart_24-03-16_16-25-10-385 (1) (1).jpg"
tags = inference_fp16(image_path, model_fp16)
print("Image Tags:", tags)











def find_missing_bbox(ocr, ocr_b):
    for i, (text, bbox) in enumerate(zip(ocr, ocr_b + [None])):
        if bbox is None:
            return i, text
    return None, None

index, missing_text = find_missing_bbox(ocr, ocr_b)

if index is not None:
    print(f"The OCR result at index {index} with text '{missing_text}' doesn't have a corresponding bounding box.")
else:
    print("All OCR results have corresponding bounding boxes.")
def remove_ocr_without_bbox(ocr, ocr_b):
    # Ensure ocr and ocr_b are lists
    ocr = list(ocr)
    ocr_b = list(ocr_b)
    
    # Find the index of the OCR result to remove
    index_to_remove = len(ocr_b)
    
    # Remove the OCR result without a bounding box
    removed_text = ocr.pop(index_to_remove)
    
    print(f"Removed OCR result: '{removed_text}'")
    print(f"New OCR list length: {len(ocr)}")
    print(f"OCR_b list length: {len(ocr_b)}")
    
    return ocr

# Call the function to remove the OCR without a bounding box
ocr = remove_ocr_without_bbox(ocr, ocr_b)









# Define the filter ranges for each position
filter_ranges = {
    "top_right_corner": (0, 0, 426, 240),
    "top_left_corner": (852, 0, 1278, 240),
    "bottom_right_corner": (0, 480, 426, 720),
    "bottom_left_corner": (852, 480, 1278, 720),
    "top_middle_side": (426, 0, 852, 240),
    "bottom_middle_side": (426, 480, 852, 720),
    "right_middle_side": (0, 240, 426, 480),
    "left_middle_side": (852, 240, 1278, 480),
    "center_point": (426, 240, 852, 480)
}
len(ocr),len(icon),len(object_)
len(ocr_b),len(icon_b),len(object_b)


all_object= ocr+icon+object_
all_b = ocr_b+icon_b+object_b
len(all_object),len(all_b)

from PIL import Image, ImageDraw, ImageFont
import random
len(all_object)
print(f"total objects before {len(all_object)}")
# Corrected filter ranges for 1920x1080
filter_ranges = {
    "top_left_corner": (0, 0, 639, 360),
    "top_right_corner": (1280, 0, 1919, 360),
    "bottom_left_corner": (0, 720, 639, 1080),
    "bottom_right_corner": (1280, 720, 1919, 1080),
    "top_middle_side": (639, 0, 1280, 360),
    "bottom_middle_side": (639, 720, 1280, 1080),
    "left_middle_side": (0, 360, 639, 720),
    "right_middle_side": (1280, 360, 1919, 720),
    "center_point": (639, 360, 1280, 720)
}



def is_inside(bbox, filter_range):
    x1, y1, x2, y2 = bbox
    fx1, fy1, fx2, fy2 = filter_range
    return (fx1 <= x1 < fx2 and fy1 <= y1 < fy2) and (fx1 <= x2 <= fx2 and fy1 <= y2 <= fy2)

def draw_bbox_and_label(draw, bbox, label, color):
    x1, y1, x2, y2 = map(int, bbox)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    font = ImageFont.load_default()
    label_size = draw.textsize(label, font=font)
    
    draw.rectangle([x1, y1 - label_size[1] - 5, x1 + label_size[0], y1], fill=color)
    draw.text((x1, y1 - label_size[1] - 5), label, fill=(255, 255, 255), font=font)



# Initialize dictionaries to store filtered objects and bounding boxes
filtered_results = {pos: {"objects": [], "bboxes": []} for pos in filter_ranges}

# Filter objects and bounding boxes
for obj, bbox in zip(all_object, all_b):
    for position, filter_range in filter_ranges.items():
        if is_inside(bbox, filter_range):
            filtered_results[position]["objects"].append(obj)
            filtered_results[position]["bboxes"].append(bbox)
            break  # Stop after finding the first matching position
total = 0
# Print results
for position, data in filtered_results.items():
    print(f"{position}:")
    total+=len(data['objects'])
    print(f"  Objects: {len(data['objects'])}")
    print(f"  Bounding Boxes: {len(data['bboxes'])}")
    print(f"  Objects: {data['objects']}")
#     print(f"  Bounding Boxes: {data['bboxes']}")
    print()
print(f"total objects after {total}")









from PIL import Image, ImageDraw, ImageFont
import random

print(f"total objects before {len(all_object)}")

# Corrected filter ranges for 1920x1080
filter_ranges = {
    "top_left_corner": (0, 0, 640, 360),
    "top_right_corner": (1280, 0, 1920, 360),
    "bottom_left_corner": (0, 720, 640, 1080),
    "bottom_right_corner": (1280, 720, 1920, 1080),
    "top_middle_side": (640, 0, 1280, 360),
    "bottom_middle_side": (640, 720, 1280, 1080),
    "left_middle_side": (0, 360, 640, 720),
    "right_middle_side": (1280, 360, 1920, 720),
    "center_point": (640, 360, 1280, 720)
}

def is_inside(bbox, filter_range):
    x1, y1, x2, y2 = bbox
    fx1, fy1, fx2, fy2 = filter_range
    return (fx1 <= x1 < fx2 and fy1 <= y1 < fy2) or (fx1 <= x2 <= fx2 and fy1 <= y2 <= fy2)

def draw_bbox_and_label(draw, bbox, label, color):
    x1, y1, x2, y2 = map(int, bbox)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    font = ImageFont.load_default()
    label_size = draw.textsize(label, font=font)
    
    draw.rectangle([x1, y1 - label_size[1] - 5, x1 + label_size[0], y1], fill=color)
    draw.text((x1, y1 - label_size[1] - 5), label, fill=(255, 255, 255), font=font)

# Initialize dictionaries to store filtered objects and bounding boxes
filtered_results = {pos: {"objects": [], "bboxes": []} for pos in filter_ranges}

# Filter objects and bounding boxes
for obj, bbox in zip(all_object, all_b):
    assigned = False
    for position, filter_range in filter_ranges.items():
        if is_inside(bbox, filter_range):
            filtered_results[position]["objects"].append(obj)
            filtered_results[position]["bboxes"].append(bbox)
            assigned = True
            break  # Stop after finding the first matching position
    
    if not assigned:
        # If the object doesn't fit in any section, assign it to the nearest one
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        min_distance = float('inf')
        nearest_position = None
        
        for position, (fx1, fy1, fx2, fy2) in filter_ranges.items():
            section_center_x = (fx1 + fx2) / 2
            section_center_y = (fy1 + fy2) / 2
            distance = ((center_x - section_center_x) ** 2 + (center_y - section_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_position = position
        
        filtered_results[nearest_position]["objects"].append(obj)
        filtered_results[nearest_position]["bboxes"].append(bbox)

total = 0
# Print results
for position, data in filtered_results.items():
    print(f"{position}:")
    total += len(data['objects'])
    print(f"  Objects: {len(data['objects'])}")
    print(f"  Bounding Boxes: {len(data['bboxes'])}")
    print(f"  Objects: {data['objects']}")
    print()

print(f"total objects after {total}")












from PIL import Image, ImageDraw, ImageFont
import random

def draw_bbox_and_label(draw, bbox, label, color):
    x1, y1, x2, y2 = map(int, bbox)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    
    # Use default font
    font = ImageFont.load_default()
    
    # Get label size
    label_size = draw.textsize(label, font=font)
    
    # Draw filled rectangle for label background
    draw.rectangle([x1, y1 - label_size[1] - 5, x1 + label_size[0], y1], fill=color)
    
    # Put text on the filled rectangle
    draw.text((x1, y1 - label_size[1] - 5), label, fill=(255, 255, 255), font=font)

# Load and resize the image
image_path = main_path
image = Image.open(image_path)
image = image.resize((1920, 1080))

# Create a draw object
draw = ImageDraw.Draw(image)
test = "left_middle_side"
test = "center_point"
test = "bottom_left_corner"
# test = "top_middle_side"
# test = "top_right_corner"
# Get data for top_left_corner
objects = filtered_results[f"{test}"]["objects"]
bboxes = filtered_results[f"{test}"]["bboxes"]

# Draw bounding boxes and labels for top_left_corner
for obj, bbox in zip(objects, bboxes):
    print(obj)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw_bbox_and_label(draw, bbox, obj, color)

# Save the image
output_path = f"{home}/output_image_with_bbox_top_left.png"
image.save(output_path)
print(f"Image saved to {output_path}")

# Display the image (this will work in a Jupyter notebook)
display(image)















def load_image(image):
    image = np.asarray(image)
    image_tensor = F.to_tensor(image)
    return image, image_tensor

def dahwin_object(image_list,threshold):
    text_prompt = []
    import re
    for idx, img in enumerate(image_list):
        tags = inference_fp16(img, model_fp16)
        tag_list = re.findall(r'\w+(?:\s+\w+)*', tags)
        text_prompt.append(tag_list)
    print(text_prompt)

    # Batch processing
    images_tensors = []
    for image_path in image_list:
        _, image_tensor = load_image(image_path)
        image_tensor = image_tensor.to(device=device).to(dtype=getattr(torch, dtype))
        images_tensors.append(image_tensor)
    image_tensor_batch = torch.stack(images_tensors)
    
    box_threshold = threshold
    text_threshold = threshold
    iou_threshold = 0.5
    import time
    s = time.time()
    with torch.no_grad():
        bbox_batch, conf_batch, class_id_batch = grounding_dino_model(
            image_batch=image_tensor_batch,
            text_prompts=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=iou_threshold
        )
        
        flattened_bbox_batch = [bbox.tolist() for bbox in bbox_batch]
        class_id_batch = [class_ids.tolist() for class_ids in class_id_batch]
#         print(len(flattened_bbox_batch[0]), len(class_id_batch[0]))
#         print(class_id_batch)
#         print(flattened_bbox_batch)
        
        # Remove None values and corresponding bounding boxes
        cleaned_class_id = []
        cleaned_bbox = []
        for t, cls, bbox in zip(text_prompt, class_id_batch, flattened_bbox_batch):
            valid_indices = [i for i, c in enumerate(cls) if c is not None]
            cleaned_class_id.append([t[cls[i]] for i in valid_indices])
            cleaned_bbox.append([bbox[i] for i in valid_indices])

    e = time.time()
    l = e - s
    print(l)
    
    object_ = cleaned_class_id[0]
    object_b = cleaned_bbox[0]
    
    answer_d = [object_, object_b,text_prompt]
    return answer_d
img = Image.open("dubu.png").convert("RGB")
# img = Image.open(f"{home}/subscribe.png").convert("RGB")
threshold = 0.15
object_,object_b,text_prompt  = dahwin_object([img],threshold)
# object_b = scale_bboxes(object_b, original_size, target_size)















from PIL import Image, ImageDraw, ImageFont
import random

# Example image and objects
image = img  # Replace with your actual image object

# Prepare to draw
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Use default font; you can specify a custom one

# Example extracted data (replace these with actual values)
class_id = object_  # List of object names, e.g., ['smile', 'girl', 'hand', ...]
flattened_bbox_batch = object_b  # List of bounding boxes, e.g., [(x0, y0, x1, y1), ...]

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Iterate and draw bounding boxes with labels
for i in range(len(class_id)):
    # Get class and box coordinates
    class_name = class_id[i]
    x0, y0, x1, y1 = flattened_bbox_batch[i]

    # Generate a random color for each bounding box
    color = random_color()

    # Draw bounding box
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=2)

    # Create a background for the text (rectangle behind text)
    text = class_name
    text_width, text_height = draw.textsize(text, font=font)  # Get text size
    padding = 5  # Add some padding around the text
    background_box = [(x0, y0 - text_height - padding), (x0 + text_width + padding, y0)]  # Box coordinates
    draw.rectangle(background_box, fill=color)  # Draw the background box behind the text

    # Draw the label with text on top of the background
    text_position = (x0 + padding // 2, y0 - text_height - padding // 2)  # Adjust position
    draw.text(text_position, text, fill="white", font=font)  # Draw the label in white text

display(image)





import os,sys
os.path.dirname(sys.executable)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
BASE_DIR = "./"
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


from chromadb.config import Settings
import chromadb
import torch
from chromadb.utils import embedding_functions


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
model_name="jinaai/jina-embeddings-v3", device="cuda:1",trust_remote_code=True)


# Use the in-memory client
chroma_client = chromadb.Client()



ef._model.half()







import torch
from sentence_transformers import SentenceTransformer

# Load model in FP16 (half precision)
model = SentenceTransformer(
    "jinaai/jina-embeddings-v3", 
    trust_remote_code=True,
    device="cuda",  # Ensure that you are using a GPU
)
model = model.half()

from sentence_transformers import SentenceTransformer, util
# List of objects
objects = [
    "chair",
    "human",
    "hen",
    "pen",
    "logo",
    "lamp",
    "pillow",
    "book",
    "elephent trunk",
    "mug",
    "laptop",
    "phone",
    "bottle",
    'microsoft windows icon'
    "keyboard",
    "shoe",
    "wallet",
    "glasses",
    "elephent",
    "guitar",
    "clock",
    "plant",
    "spoon"
]

# Target word for similarity check
target_word = "animale"

# Encode the target word and objects
target_embedding = model.encode([target_word], convert_to_tensor=True)
object_embeddings = model.encode(objects, convert_to_tensor=True)

# Compute cosine similarity
cosine_scores = util.cos_sim(target_embedding, object_embeddings)[0]

# Filter objects based on a higher similarity threshold (e.g., 0.6 or 0.7)
threshold = 0.5
related_objects = [obj for obj, score in zip(objects, cosine_scores) if score >= threshold]

print("Related objects:", related_objects)








def return_bbox(checked_object):
        
        
        objj =  [
    "top_left_corner",
    "top_right_corner",
    "bottom_left_corner",
    "bottom_right_corner",
    "top_middle_side",
    "bottom_middle_side",
    "left_middle_side",
    "right_middle_side",
    "center_point"
]



        import time
        s = time.time()


        try:
            chroma_client.delete_collection(name="dahyuendahwin")
        except:
            pass

        collection = chroma_client.create_collection(name="dahyuendahwin",embedding_function=ef)
        # Generate unique IDs for each object
        ids = [f"id{i+1}" for i in range(len(objj))]

        # Upsert the objects into the collection
        collection.upsert(
            documents=objj,
            ids=ids
        )
        results = collection.query(
        #     query_texts=[""""Click on the Microsoft Edge icon to open the web browser. top_left_corner"""], # Chroma will embed this for you
            query_texts=[f""""{checked_object}"""], # Chroma will embed this for you
            n_results=4 # how many results to return
        )

        # print(results)
        res = results['documents'][0][0]
        print(f'res {res}')
        return res
checked_object = "search icon middel of the top tier"
return_bbox(checked_object)












from PIL import Image
import random

# Set the dimensions for a Full HD image (1920x1080)
width = 1920
height = 1080

# Create a new black image
black_image = Image.new('RGB', (width, height), color='black')

# Open the mouse icon
mouse_icon_path = f"{home}/mouse_icon.png"

mouse_icon = Image.open(mouse_icon_path)

# Resize the mouse icon
icon_size = (20, 20)  # New size for the icon
mouse_icon_resized = mouse_icon.resize(icon_size, Image.LANCZOS)

# Set up row parameters
num_rows = 15
min_spacing = 5  # Minimum space between icons

# Function to generate non-overlapping x-positions for a row
def generate_x_positions(num_mice, min_spacing):
    positions = []
    while len(positions) < num_mice:
        new_pos = random.randint(0, width - icon_size[0])
        if all(abs(new_pos - pos) >= (icon_size[0] + min_spacing) for pos in positions):
            positions.append(new_pos)
    return positions

# Paste scattered rows of mouse icons
for _ in range(num_rows):
    # Randomly choose number of mice for this row (2 to 10)
    num_mice = random.randint(2, 10)
    
    # Generate random y position for this row
    y_position = random.randint(0, height - icon_size[1])
    
    # Generate non-overlapping x positions for this row
    x_positions = generate_x_positions(num_mice, min_spacing)
    
    # Paste the mice for this row
    for x_position in x_positions:
        black_image.paste(mouse_icon_resized, (x_position, y_position), mouse_icon_resized)
black_image.save(f"{home}/black.png")
# Save the image
display(black_image)
















import cv2
import numpy as np
import time
img_path = main_path
def draw_bbox_and_label(img, bbox, label, color):
    """
    Draw bounding box and label using OpenCV with smaller font
    
    Args:
        img: OpenCV image (numpy array)
        bbox: tuple/list of (x1, y1, x2, y2)
        label: string label to display
        color: tuple of (B, G, R) for OpenCV
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Use smaller font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # Reduced from 0.6
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw filled rectangle for label background with smaller padding
    cv2.rectangle(img, 
                 (x1, y1 - label_height - 5),  # Reduced padding from 10 to 5
                 (x1 + label_width, y1),
                 color, 
                 -1)
    
    # Put text with adjusted position
    cv2.putText(img, 
                label,
                (x1, y1 - 3),  # Adjusted y position for better alignment
                font,
                font_scale,
                (255, 255, 255),
                thickness)
    
    return img

def process_image_with_bboxes(image_path, objects, bboxes,printt=None, output_path=None,):
    """
    Process image and draw all bounding boxes
    
    Args:
        image_path: path to input image
        objects: list of object labels
        bboxes: list of bounding boxes (x1,y1,x2,y2)
        output_path: path to save output image (optional)
    Returns:
        processed image as numpy array
    """
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1920, 1080))
    
    # Process each bbox
    for obj, bbox in zip(objects, bboxes):
        # if printt!=None:
        #     print(obj)
        # Random BGR color
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        img = draw_bbox_and_label(img, bbox, obj, color)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, img)
        
    return img

test = "left_middle_side"
test = "center_point"
# test = "bottom_left_corner"
# test = "top_right_corner"
test = "top_middle_side"
test = 'right_middle_side'
# test = "top_right_corner"
test = "top_left_corner"
# Get data for top_left_corner
objects = filtered_results[f"{test}"]["objects"]
bboxes = filtered_results[f"{test}"]["bboxes"]



start = time.time()


img = process_image_with_bboxes(
    img_path,
    all_object,
    all_b,      
    output_path
)



printt=None
# img = process_image_with_bboxes(
#     img_path,
#     objects,  # Your list of object labels
#     bboxes,       # Your list of bounding boxes
#         printt,
#     output_path,

# )

end = time.time()
print(f"Processing time: {end - start:.4f} seconds")

# Convert the color format from BGR to RGB
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (NumPy array) to a PIL image
pil_image = Image.fromarray(image_rgb)
pil_image

display(pil_image)










    
def return_bbox(ai_output, filtered_results, action,n,all_object,all_b):
        def find_nth_occurrence(data, element, n):
            count = 0  # Counter for occurrences
            last = None
            for i, value in enumerate(data):
                if value == element:
                    count += 1
                    last=i
                    if count == n:
                        return i  # Return the index of the nth occurrence
            return last  
        global history, done
        import re
        # Define the pattern to match the desired fields
        pattern = re.compile(r'(?i)(<filter_range>|<checked_object>|<yestyping>)\s*:\s*(.*)')
        # Find all matches in the input string
        matches = pattern.findall(ai_output)
        # Create a dictionary to store the extracted information
        extracted_info = {key.lower(): value.strip() for key, value in matches}
        filter_range = extracted_info.get('<filter_range>', None)
        checked_object = extracted_info.get('<checked_object>', None)
        yestyping = extracted_info.get('<yestyping>', None)
        # Print the extracted information
        print("Extracted information:")
        print("filter_range:", filter_range)
        print("checked_object:", checked_object)
        print("yestyping:", yestyping)
        
        s_action_history = f"{action} {checked_object}"
        
        print(f'history : {s_action_history}')
        second_action_history.append(s_action_history)        
        if yestyping != None:
            s_action_history = f"typed: {yestyping}"
            second_action_history.append(s_action_history)
        if 'full' !=filter_range:
            # Get data for top_left_corner
            objects = filtered_results[f"{filter_range}"]["objects"]
            bboxes = filtered_results[f"{filter_range}"]["bboxes"]
            objj = []
            bboxx = []
            
            # Create a list of tuples containing (object, bbox, y_coord, x_coord)
            sorted_items = []
            for obj, bbox in zip(objects, bboxes):
                x_min, y_min, x_max, y_max = bbox
                # Use y_min as primary sort key (top to bottom)
                # Use x_min as secondary sort key (left to right)
                sorted_items.append((obj, bbox, y_min, x_min))
            
            # Sort based on y_coordinate first, then x_coordinate
            # Using a threshold to group items that are roughly on the same line
            threshold = 20  # Adjust this value based on your needs
            
            # Sort primarily by y-coordinate with threshold grouping, then by x-coordinate
            sorted_items.sort(key=lambda x: (x[2] // threshold, x[3]))
            
            # Unzip the sorted items back into separate lists
            objj, bboxx, _, _ = zip(*sorted_items)
            objj = list(objj)
            bboxx = list(bboxx)
        else:
            objj = all_object
            bboxx = all_b

        import time
        s = time.time()
        try:
            chroma_client.delete_collection(name="dahyuendahwin")
        except:
            pass
        collection = chroma_client.create_collection(name="dahyuendahwin",embedding_function=ef)
        # Generate unique IDs for each object
        ids = [f"id{i+1}" for i in range(len(objj))]
        # Upsert the objects into the collection
        collection.upsert(
            documents=objj,
            ids=ids
        )
        results = collection.query(
            query_texts=[f""""{checked_object}"""], # Chroma will embed this for you
            n_results=4 # how many results to return
        )
        print(f'all results {results}')

        res = [results['documents'][0]]
        print(f'pre res {res}')
        
        yes = next((item for i in res for item in i if checked_object in item), None)
        if yes==None:
            print('yes none')
            yes = next((item for i in res for item in i if checked_object.lower() in item.lower()), None)
        if yes!=None:
            res = yes
            print(f'yes {yes}')
            res = find_nth_occurrence(objj, res, n)
            bbox = bboxx[res]

            full_text = yes
            segment_text = checked_object


            if ' ' in segment_text:
                
                try:
                    new_bbox = get_text_segment_bbox_space(img_path, full_text, segment_text, bbox)
                except:
                    new_bbox = get_text_segment_bbox_nospace(img_path, full_text, segment_text, bbox) 
            else:
                try:
                    new_bbox = get_text_segment_bbox_nospace(img_path, full_text, segment_text, bbox) 
                except:
                    new_bbox = get_text_segment_bbox_space(img_path, full_text, segment_text, bbox)
                    
            bbox = new_bbox
                        
        else:
            res = results['documents'][0][0]
            res = find_nth_occurrence(objj, res, n)
            bbox = bboxx[res]
        
        
        print(f'res {res}')
#         res = objj.index(res)

        print(bbox)
        e = time.time()
        r = e-s
        print(f"total time {r}")
        x_min, y_min, x_max, y_max = bbox
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        text = f'x={x_center}, y={y_center}'
        print(text)
        return text, bbox









import cv2
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from numpy import random
import os
import cv2
import re
import httpx
import asyncio
import aiohttp
import nest_asyncio
import time
from mistralai import Mistral
import threading
import difflib

second_action_history = []

import math

def calculate_distance(box1, box2):
    # Calculate the center of each box
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2

    # Return the Euclidean distance between the centers
    return math.dist([x1_center, y1_center], [x2_center, y2_center])

def find_closest_bbox(heading, all_bbox, direction=None):
    closest_box = None
    min_distance = float('inf')

    for bbox in all_bbox:
        x1_min, y1_min, x1_max, y1_max = heading
        x2_min, y2_min, x2_max, y2_max = bbox

        if direction == "top" and y2_max >= y1_min:
            continue
        elif direction == "bottom" and y2_min <= y1_max:
            continue
        elif direction == "left" and x2_max >= x1_min:
            continue
        elif direction == "right" and x2_min <= x1_max:
            continue

        distance = calculate_distance(heading, bbox)
        if distance < min_distance:
            min_distance = distance
            closest_box = bbox

    return closest_box


def find_missing_bbox(ocr, ocr_b):
    for i, (text, bbox) in enumerate(zip(ocr, ocr_b + [None])):
        if bbox is None:
            return i, text
    return None, None

index, missing_text = find_missing_bbox(ocr, ocr_b)

def remove_ocr_without_bbox(ocr, ocr_b):
    if ocr!=[]:
        # Ensure ocr and ocr_b are lists
        ocr = list(ocr)
        ocr_b = list(ocr_b)

        # Find the index of the OCR result to remove
        index_to_remove = len(ocr_b)

        # Remove the OCR result without a bounding box
        removed_text = ocr.pop(index_to_remove)

    #     print(f"Removed OCR result: '{removed_text}'")
    #     print(f"New OCR list length: {len(ocr)}")
    #     print(f"OCR_b list length: {len(ocr_b)}")

        return ocr
    else:
        return []

def get_text_segment_bbox_space(img, full_text, segment_text, original_bbox, color=(0, 0, 255), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    print("Initial values:")
    print(f"full_text: '{full_text}'")
    print(f"segment_text: '{segment_text}'")
    
    # Load the image
    img = cv2.imread(img)
    if 'icon' not in full_text.lower():

        # First try exact match
        start_index = full_text.find(segment_text)
        if start_index == -1:
            # If not found, try case-insensitive
            full_text_lower = full_text.lower()
            segment_text_lower = segment_text.lower()
            start_index = full_text_lower.find(segment_text_lower)
            if start_index == -1:
                raise ValueError("Segment text not found in full text")

        # Calculate the substring before the target segment
        prefix_text = full_text[:start_index]

        # Calculate the font scale that was used for the original bbox
        original_width = original_bbox[2] - original_bbox[0]
        font_scale = 0.1
        max_scale = 100
        tolerance = 2

        while font_scale < max_scale:
            (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)

            if abs(text_width - original_width) < tolerance:
                break
            elif text_width < original_width:
                font_scale *= 1.1
            else:
                font_scale *= 0.9

        # Calculate width of the prefix text
        (prefix_width, _), _ = cv2.getTextSize(prefix_text, font, font_scale, thickness)

        # Calculate width of the segment
        (segment_width, segment_height), baseline = cv2.getTextSize(segment_text, font, font_scale, thickness)

        # Calculate new bbox coordinates
        new_x1 = original_bbox[0] + prefix_width
        new_x2 = new_x1 + segment_width
        new_y1 = original_bbox[1]
        new_y2 = original_bbox[3]

        new_bbox = [int(new_x1), new_y1, int(new_x2), new_y2]
        
    else:
        new_bbox = original_bbox


    
    return new_bbox

def get_text_segment_bbox_nospace(img, full_text, segment_text, original_bbox, color=(0, 0, 255), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    print("Initial values:")
    print(f"full_text: '{full_text}'")
    print(f"segment_text: '{segment_text}'")
    # Load the image
    img = cv2.imread(img)
    # if segment_text not in full_text:
    #     raise ValueError("Segment text must be part of the full text")
    
    if 'icon' not in full_text.lower():
        try:
            # Split the text into words and find the target word
            words = full_text.split()
            target_word_index = words.index(segment_text)
        except:
            print('except')
            try:

                full_text = full_text.lower()
                segment_text = segment_text.lower()
                # Split the text into words and find the target word
                words = full_text.split()
                target_word_index = words.index(segment_text)
            except:

                query =segment_text
                # Step 1: Find the closest match without any filtering
                closest_match = difflib.get_close_matches(query, words, n=1, cutoff=0.0)

                # Step 2: Check if the closest match meets the length condition
                if closest_match and len(closest_match[0]) <= len(query):
                    target_word_index =words.index(closest_match[0])
                else:
                    # Step 3: Filter the list and find the closest match again
                    filtered_text = [word for word in words if len(word) <= len(query)]
                    new_closest_match = difflib.get_close_matches(query, filtered_text, n=1, cutoff=0.0)
                    target_word_index =words.index(new_closest_match[0])





        # Calculate the font scale that was used for the original bbox
        original_width = original_bbox[2] - original_bbox[0]
        font_scale = 0.1
        max_scale = 100
        tolerance = 2

        while font_scale < max_scale:
            (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)

            if abs(text_width - original_width) < tolerance:
                break
            elif text_width < original_width:
                font_scale *= 1.1
            else:
                font_scale *= 0.9

        # Calculate space width
        (space_width, _), _ = cv2.getTextSize(" ", font, font_scale, thickness)

        # Calculate width up to our target word
        prefix_width = 0
        for i in range(target_word_index):
            (word_width, _), _ = cv2.getTextSize(words[i], font, font_scale, thickness)
            prefix_width += word_width + space_width

        # Get the width of our segment
        (segment_width, segment_height), baseline = cv2.getTextSize(segment_text, font, font_scale, thickness)

        # Calculate new bbox coordinates
        new_x1 = original_bbox[0] + prefix_width
        new_x2 = new_x1 + segment_width
        new_y1 = original_bbox[1]
        new_y2 = original_bbox[3]

        new_bbox = [int(new_x1), new_y1, int(new_x2), new_y2]
    else:
        new_bbox = original_bbox


    return new_bbox


    
    
def scale_bbox(bbox, scale_factor=1.1):
    """
    Scale a bounding box by a specified factor.
    Keeps the bbox centered while scaling.
    
    :param bbox: The original bounding box (x_min, y_min, x_max, y_max).
    :param scale_factor: The factor by which to scale the bounding box.
    :return: The scaled bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the amount to expand on each side
    delta_width = (width * (scale_factor - 1)) / 2
    delta_height = (height * (scale_factor - 1)) / 2

    # Apply expansion
    x_min_new = max(0, int(x_min - delta_width))
    y_min_new = max(0, int(y_min - delta_height))
    x_max_new = int(x_max + delta_width)
    y_max_new = int(y_max + delta_height)

    return x_min_new, y_min_new, x_max_new, y_max_new

    
def return_bbox(ai_output,direction, filtered_results, action,n,all_object,all_b):
        def find_nth_occurrence(data, element, n):
            count = 0  # Counter for occurrences
            last = None
            for i, value in enumerate(data):
                if value == element:
                    count += 1
                    last=i
                    if count == n:
                        return i  # Return the index of the nth occurrence
            return last  
        global history, done,bbox
        import re
        pre = []
        # Define the pattern to match the desired fields
        pattern = re.compile(r'(?i)(<filter_range>|<checked_object>|<close>|<yestyping>)\s*:\s*(.*)')
        # Find all matches in the input string
        matches = pattern.findall(ai_output)
        # Create a dictionary to store the extracted information
        extracted_info = {key.lower(): value.strip() for key, value in matches}
        filter_range = extracted_info.get('<filter_range>', None)
        checked_object = extracted_info.get('<checked_object>', None)
        close = extracted_info.get('<close>', None)
        yestyping = extracted_info.get('<yestyping>', None)
        # Print the extracted information
        print("Extracted information:")
        print("filter_range:", filter_range)
        print("checked_object:", checked_object)
        print('<close>:',close)
        print("yestyping:", yestyping)
        
        s_action_history = f"{action} {checked_object}"
        
        print(f'history : {s_action_history}')
        second_action_history.append(s_action_history)
        
        
        
        
        
        
        
        
        if yestyping != None:
            s_action_history = f"typed: {yestyping}"
            second_action_history.append(s_action_history)
        if 'full' !=filter_range:
            # Get data for top_left_corner
            objects = filtered_results[f"{filter_range}"]["objects"]
            bboxes = filtered_results[f"{filter_range}"]["bboxes"]
            objj = []
            bboxx = []
            
            # Create a list of tuples containing (object, bbox, y_coord, x_coord)
            sorted_items = []
            for obj, bbox in zip(objects, bboxes):
                x_min, y_min, x_max, y_max = bbox
                # Use y_min as primary sort key (top to bottom)
                # Use x_min as secondary sort key (left to right)
                sorted_items.append((obj, bbox, y_min, x_min))
            
            # Sort based on y_coordinate first, then x_coordinate
            # Using a threshold to group items that are roughly on the same line
            threshold = 20  # Adjust this value based on your needs
            
            # Sort primarily by y-coordinate with threshold grouping, then by x-coordinate
            sorted_items.sort(key=lambda x: (x[2] // threshold, x[3]))
            
            # Unzip the sorted items back into separate lists
            objj, bboxx, _, _ = zip(*sorted_items)
            objj = list(objj)
            bboxx = list(bboxx)
        else:
            objj = all_object
            bboxx = all_b

        import time
        
        
        
        if close!=None:
            checked_object_list = [close,checked_object]
        else:
            checked_object_list = [checked_object]
        for checked_object in checked_object_list:
            

            s = time.time()
            try:
                chroma_client.delete_collection(name="dahyuendahwin")
            except:
                pass
            collection = chroma_client.create_collection(name="dahyuendahwin",embedding_function=ef)
            # Generate unique IDs for each object
            ids = [f"id{i+1}" for i in range(len(objj))]
            # Upsert the objects into the collection
            collection.upsert(
                documents=objj,
                ids=ids
            )
            results = collection.query(
                query_texts=[f""""{checked_object}"""], # Chroma will embed this for you
                n_results=4 # how many results to return
            )
            print(f'all results {results}')









            res = [results['documents'][0]]

            all_res_bbox = []
            for i in res:
                for item in i:
                    print(item)

                    r = find_nth_occurrence(objj,item, n)
                    r= bboxx[r]
                    print(r)
                    all_res_bbox.append(r)
            print(f'full bbox {all_res_bbox}')
            
            
#             count_dict = {}
#             for i in res:
#                 for item in i:
#                     if item in count_dict:
#                         count_dict[item] += 1
#                     else:
#                         count_dict[item] = 1
#                     n = count_dict[item]


#                     r = find_nth_occurrence(objj,item, n)
#                     r= bboxx[r]
#                     print(r)
#                     all_res_bbox.append(r)
#             print(f'full bbox {all_res_bbox}')



            print(f'pre res {res}')

            if pre!=[]:
                yes=  find_closest_bbox(pre[-1], all_res_bbox, direction=direction)
                print(yes)
                if yes!=None:
                    idx = all_res_bbox.index(yes)
                    yes = res[0][idx]
            else:
                yes = next((item for i in res for item in i if checked_object in item), None)
                if yes==None:
                    print('yes none')
                    yes = next((item for i in res for item in i if checked_object.lower() in item.lower()), None)
            if yes!=None:
                res = yes
                print(f'yes {yes}')
                res = find_nth_occurrence(objj, res, n)
                bbox = bboxx[res]

                full_text = yes
                segment_text = checked_object


                if ' ' in segment_text:

                    try:
                        print('space')
                        new_bbox = get_text_segment_bbox_space(img_path, full_text, segment_text, bbox)
                    except:
                        print('nospace')
                        new_bbox = get_text_segment_bbox_nospace(img_path, full_text, segment_text, bbox) 
                else:
                    try:
                        print('else nospace')
                        new_bbox = get_text_segment_bbox_nospace(img_path, full_text, segment_text, bbox) 
                    except:
                        print('else space')
                        new_bbox = get_text_segment_bbox_space(img_path, full_text, segment_text, bbox)

                bbox = new_bbox

            else:
                res = results['documents'][0][0]
                res = find_nth_occurrence(objj, res, n)
                bbox = bboxx[res]


            print(f'res {res}')
            pre.append(bbox)
            print(bbox)
        e = time.time()
        r = e-s
        print(f"total time {r}")
        bbox = scale_bbox(bbox, scale_factor=1.2)
        x_min, y_min, x_max, y_max = bbox
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        text = f'x={x_center}, y={y_center}'
        print(text)
        
        return text, bbox
    
    


    
    
def filtered_result(frame):
    # Get the width and height of the image
    height, width, _ = frame.shape
    


    
    # Convert the frame to RGB format (OpenCV uses BGR by default)
    frame_rgb = frame

    # Convert the NumPy array (frame) to a PIL Image
    pil_image = Image.fromarray(frame_rgb)
    original_size = pil_image.size
    

    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Print the aspect ratio
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    # Check if the aspect ratio is within the specified range
    if 1.76 <= aspect_ratio <= 1.79:
        target_size = (1920, 1080)
        img_resized = cv2.resize(frame_rgb, target_size)
        img_list = [img_resized]
        img = pil_image.resize((1280, 720))

        print(True)
    else:
        target_size = (1920, 1080)
        img_resized = cv2.resize(frame_rgb, target_size)
        img_list = [img_resized]
        img = pil_image


        print(False)






    try:
        ocr_b, ocr = dahwin_ocr(img_list)
    except:
        ocr_b=[]
        ocr = []
    
    # Call the function to remove the OCR without a bounding box
    ocr = remove_ocr_without_bbox(ocr, ocr_b)
    # print(ocr)
    # print(ocr_b)


    

    try:
        bounding_boxes,clss = dahwin_icon([pil_image])
    except:
        bounding_boxes=[]
        clss = []
    icon = clss
    # print(icon)

    
    icon_b = scale_bboxes(bounding_boxes, original_size, target_size)
    # print(icon_b)


    try:
         object_,object_b = dahwin_object([img])
    except:
        object_=[]
        object_b =[]
    original_size = img.size
    object_b = scale_bboxes(object_b, original_size, target_size)

    

    if len(ocr) != len(ocr_b) and len(icon) != len(icon_b) and len(object_) != len(object_b):
        raise AssertionError("Lengths of ocr, icon, and object variables do not match their corresponding '_b' variables.")

        
    all_object= ocr+icon+object_
    all_b = ocr_b+icon_b+object_b
        
    filtered_results = filter(all_object,all_b)
    

        



    test = "top_left_corner"

    top_left_corner = filtered_results[f"{test}"]["objects"]

    test = "top_right_corner"

    top_right_corner = filtered_results[f"{test}"]["objects"]


    test = "bottom_left_corner"

    bottom_left_corner = filtered_results[f"{test}"]["objects"]

    test = "bottom_right_corner"

    bottom_right_corner = filtered_results[f"{test}"]["objects"]

    test = "top_middle_side"

    top_middle_side = filtered_results[f"{test}"]["objects"]

    test = "bottom_middle_side"

    bottom_middle_side = filtered_results[f"{test}"]["objects"]

    test = "left_middle_side"

    left_middle_side = filtered_results[f"{test}"]["objects"]

    test = "right_middle_side"

    right_middle_side = filtered_results[f"{test}"]["objects"]

    test = "center_point"

    center_point = filtered_results[f"{test}"]["objects"]
    


    return top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b
        
        
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from queue import Queue
import cv2
from PIL import Image

def filtered_result(frame,threshold):
    # Get the width and height of the image
    height, width, _ = frame.shape
    


    
    # Convert the frame to RGB format (OpenCV uses BGR by default)
    frame_rgb = frame

    # Convert the NumPy array (frame) to a PIL Image
    pil_image = Image.fromarray(frame_rgb)
    original_size = pil_image.size
    

    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Print the aspect ratio
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    # Check if the aspect ratio is within the specified range
    if 1.76 <= aspect_ratio <= 1.79:
        target_size = (1920, 1080)
        img_resized = cv2.resize(frame_rgb, target_size)
        img_list = [img_resized]
        img = pil_image.resize((1280, 720))

        print(True)
    else:
        target_size = original_size
        img_resized = cv2.resize(frame_rgb, target_size)
        img_list = [img_resized]
        img = pil_image


        print(False)
    
    # Create queues to store results from each thread
    ocr_queue = Queue()
    icon_queue = Queue()
    object_queue = Queue()
    
    def process_ocr():
        try:
            ocr_b, ocr = dahwin_ocr(img_list)
            # Remove OCR without bounding box
            ocr = remove_ocr_without_bbox(ocr, ocr_b)
            ocr_queue.put((ocr_b, ocr))
        except:
            ocr_queue.put(([], []))
            
    def process_icon():
        try:
            bounding_boxes, clss = dahwin_icon([pil_image])
            icon_b = scale_bboxes(bounding_boxes, pil_image.size, target_size)
            icon_queue.put((icon_b, clss))
        except:
            icon_queue.put(([], []))
            
    def process_object():
        try:
            object_, object_b,text_prompt = dahwin_object([img],threshold)
            object_b = scale_bboxes(object_b, img.size, target_size)
            object_queue.put((object_, object_b,text_prompt))
        except:
            object_queue.put(([], []))

    # Create and start threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        ocr_future = executor.submit(process_ocr)
        icon_future = executor.submit(process_icon)
        object_future = executor.submit(process_object)
        
        # Wait for all threads to complete
        ocr_future.result()
        icon_future.result()
        object_future.result()
    
    # Get results from queues
    ocr_b, ocr = ocr_queue.get()
    icon_b, icon = icon_queue.get()
    object_, object_b,text_prompt = object_queue.get()

    # Validate lengths
    if len(ocr) != len(ocr_b) or len(icon) != len(icon_b) or len(object_) != len(object_b):
        raise AssertionError("Lengths of ocr, icon, and object variables do not match their corresponding '_b' variables.")

    f = ['computer screen', 'screenshot', 'website']
    f_result = any(item in text_prompt[0] for item in f)
    if f_result:
        

        # Combine all results
        all_object = ocr + icon + object_
        all_b = ocr_b + icon_b + object_b
    else:

        # Combine all results
        all_object = ocr  + object_
        all_b = ocr_b  + object_b
    
    # Apply filtering
    filtered_results = filter(all_object, all_b)
    
    # Extract results for different regions
    regions = [
        "top_left_corner",
        "top_right_corner",
        "bottom_left_corner",
        "bottom_right_corner",
        "top_middle_side",
        "bottom_middle_side",
        "left_middle_side",
        "right_middle_side",
        "center_point"
    ]
    
    # Get objects for each region
    region_objects = [filtered_results[region]["objects"] for region in regions]
    
    # Return all results
    return (*region_objects, filtered_results, all_object, all_b)

    
    
    
# Corrected filter ranges for 1920x1080
filter_ranges = {
    "top_left_corner": (0, 0, 640, 360),
    "top_right_corner": (1280, 0, 1920, 360),
    "bottom_left_corner": (0, 720, 640, 1080),
    "bottom_right_corner": (1280, 720, 1920, 1080),
    "top_middle_side": (640, 0, 1280, 360),
    "bottom_middle_side": (640, 720, 1280, 1080),
    "left_middle_side": (0, 360, 640, 720),
    "right_middle_side": (1280, 360, 1920, 720),
    "center_point": (640, 360, 1280, 720)
}

    


                
def filter(all_object,all_b):


    def is_inside(bbox, filter_range):
        x1, y1, x2, y2 = bbox
        fx1, fy1, fx2, fy2 = filter_range
        return (fx1 <= x1 < fx2 and fy1 <= y1 < fy2) or (fx1 <= x2 <= fx2 and fy1 <= y2 <= fy2)

    def draw_bbox_and_label(draw, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        font = ImageFont.load_default()
        label_size = draw.textsize(label, font=font)

        draw.rectangle([x1, y1 - label_size[1] - 5, x1 + label_size[0], y1], fill=color)
        draw.text((x1, y1 - label_size[1] - 5), label, fill=(255, 255, 255), font=font)

    # Initialize dictionaries to store filtered objects and bounding boxes
    filtered_results = {pos: {"objects": [], "bboxes": []} for pos in filter_ranges}

    # Filter objects and bounding boxes
    for obj, bbox in zip(all_object, all_b):
        assigned = False
        for position, filter_range in filter_ranges.items():
            if is_inside(bbox, filter_range):
                filtered_results[position]["objects"].append(obj)
                filtered_results[position]["bboxes"].append(bbox)
                assigned = True
                break  # Stop after finding the first matching position

        if not assigned:
            # If the object doesn't fit in any section, assign it to the nearest one
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            min_distance = float('inf')
            nearest_position = None

            for position, (fx1, fy1, fx2, fy2) in filter_ranges.items():
                section_center_x = (fx1 + fx2) / 2
                section_center_y = (fy1 + fy2) / 2
                distance = ((center_x - section_center_x) ** 2 + (center_y - section_center_y) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    nearest_position = position

            filtered_results[nearest_position]["objects"].append(obj)
            filtered_results[nearest_position]["bboxes"].append(bbox)


    return  filtered_results   
            
            
def filter_vct(all_object,all_b):
    # Example initial code
    print(f"total objects before {len(all_object)}")

    # Corrected filter ranges for 1920x1080
    filter_ranges = {
        "top_left_corner": (0, 0, 640, 360),
        "top_right_corner": (1280, 0, 1920, 360),
        "bottom_left_corner": (0, 720, 640, 1080),
        "bottom_right_corner": (1280, 720, 1920, 1080),
        "top_middle_side": (640, 0, 1280, 360),
        "bottom_middle_side": (640, 720, 1280, 1080),
        "left_middle_side": (0, 360, 640, 720),
        "right_middle_side": (1280, 360, 1920, 720),
        "center_point": (640, 360, 1280, 720)
    }

    def calculate_intersection_area(bbox, filter_range):
        x1, y1, x2, y2 = bbox
        fx1, fy1, fx2, fy2 = filter_range

        # Calculate the overlap dimensions
        overlap_x1 = max(x1, fx1)
        overlap_y1 = max(y1, fy1)
        overlap_x2 = min(x2, fx2)
        overlap_y2 = min(y2, fy2)

        # Calculate the width and height of the intersection
        overlap_width = max(0, overlap_x2 - overlap_x1)
        overlap_height = max(0, overlap_y2 - overlap_y1)

        # Return the area of intersection
        return overlap_width * overlap_height

    def draw_bbox_and_label(draw, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        font = ImageFont.load_default()
        label_size = draw.textsize(label, font=font)

        draw.rectangle([x1, y1 - label_size[1] - 5, x1 + label_size[0], y1], fill=color)
        draw.text((x1, y1 - label_size[1] - 5), label, fill=(255, 255, 255), font=font)

    # Initialize dictionaries to store filtered objects and bounding boxes
    filtered_results = {pos: {"objects": [], "bboxes": []} for pos in filter_ranges}

    # Filter objects and bounding boxes
    for obj, bbox in zip(all_object, all_b):
        total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        for position, filter_range in filter_ranges.items():
            intersection_area = calculate_intersection_area(bbox, filter_range)

            if intersection_area > 0:
                proportion = intersection_area / total_area
                # Distribute object and bbox proportionally
                filtered_results[position]["objects"].append(obj)
                filtered_results[position]["bboxes"].append(bbox)

    # Print results
    total = 0
    for position, data in filtered_results.items():
        total += len(data['objects'])
    print(f"total objects after {total}")

    return  filtered_results   
            
    
    
    


def position(frame,threshold):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
        
#     dis = get_dis(img)


 
    
    
#     filtered_results = filter_vct(all_object,all_b)
    
    
    
    
    s = time.time()
    top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b = filtered_result(frame,threshold)
    e = time.time()
    
    print(f'total time = {e-s}')
    return top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b











# img_path = f"{home}/agi-.png"
img_path = "/kaggle/input/dataaa/black.png"
img_path = f"{home}/black.png"
# img_path = "/kaggle/input/dataaa/total_image_1.png"
img_path = "/kaggle/input/dataaa/adobe.png"
img_path = "/kaggle/input/dataaa/word.png"
# img_path = 'scrollbar.png'
img_path = 'scrollbar1.png'
# img_path ="uploaded_images/youtube.png"
img_path = f'{home}/p.png'
# img_path = f"{home}/tree0.jpg"
# img_path = 'tree.png'
# img_path = f"{home}/dubu.png"
imggg = Image.open(img_path)
import numpy as np
import cv2

numpy_image = np.array(imggg)



threshold=0.15
# Convert RGB to BGR 
frame = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b = position(frame,threshold)









import cv2
import numpy as np
import time

def draw_bbox_and_label(img, bbox, label, color):
    """
    Draw bounding box and label using OpenCV with smaller font
    
    Args:
        img: OpenCV image (numpy array)
        bbox: tuple/list of (x1, y1, x2, y2)
        label: string label to display
        color: tuple of (B, G, R) for OpenCV
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Use smaller font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # Reduced from 0.6
    thickness = 1
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw filled rectangle for label background with smaller padding
    cv2.rectangle(img, 
                 (x1, y1 - label_height - 5),  # Reduced padding from 10 to 5
                 (x1 + label_width, y1),
                 color, 
                 -1)
    
    # Put text with adjusted position
    cv2.putText(img, 
                label,
                (x1, y1 - 3),  # Adjusted y position for better alignment
                font,
                font_scale,
                (255, 255, 255),
                thickness)
    
    return img

def process_image_with_bboxes(image_path, objects, bboxes,printt=None, output_path=None,):
    """
    Process image and draw all bounding boxes
    
    Args:
        image_path: path to input image
        objects: list of object labels
        bboxes: list of bounding boxes (x1,y1,x2,y2)
        output_path: path to save output image (optional)
    Returns:
        processed image as numpy array
    """
    # Read and resize image
    img = cv2.imread(image_path)
    # img = cv2.resize(img, (1920, 1080))
    
    # Process each bbox
    for obj, bbox in zip(objects, bboxes):
        # if printt!=None:
        #     print(obj)
        # Random BGR color
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
        img = draw_bbox_and_label(img, bbox, obj, color)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, img)
        
    return img

test = "left_middle_side"
test = "center_point"
# test = "bottom_left_corner"
# test = "top_right_corner"
test = "top_middle_side"
# test = 'right_middle_side'
# test = "top_right_corner"
# test = "top_left_corner"
# Get data for top_left_corner
objects = filtered_results[f"{test}"]["objects"]
bboxes = filtered_results[f"{test}"]["bboxes"]



start = time.time()


img = process_image_with_bboxes(
    img_path,
    all_object,
    all_b,      
    output_path
)



# printt=None
# img = process_image_with_bboxes(
#     img_path,
#     objects,  # Your list of object labels
#     bboxes,       # Your list of bounding boxes
#         printt,
#     output_path,

# )

end = time.time()
print(f"Processing time: {end - start:.4f} seconds")

# Convert the color format from BGR to RGB
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (NumPy array) to a PIL image
pil_image = Image.fromarray(image_rgb)
display(pil_image)












# ai_output = """
# <filter_range>:top_right_corner
# <checked_object>: + icon
# <yestyping>:
# """
# action = "left click"
# text, bbox = return_bbox(ai_output,filtered_results,action,5)




ai_output = """
<filter_range>:top_right_corner
<checked_object>: speaker icon
<yestyping>:
"""
action = "left click"
direction = None
text, bbox = return_bbox(ai_output,    direction ,filtered_results,action,5,all_object,all_b)




# ai_output = """
# <filter_range>:center_point
# <checked_object>:hi
# <yestyping>:
# """
# action = "left click"
# text, bbox = return_bbox(ai_output,filtered_results,action,5)













from PIL import Image, ImageDraw

    

def get_points(filter_range,object,close,direction,n):
    ai_output = f"""
    <filter_range>:{filter_range}
    <checked_object>: {object}
    <yestyping>:
    """
    if close!=None:
        ai_output = f"""
        <filter_range>:{filter_range}
        <checked_object>: {object}
        <close>:{close}
        <yestyping>:
        """
    action = "left click"
    text, bbox = return_bbox(ai_output,direction,filtered_results,action,n,all_object,all_b)
    

    # Load the image
    img = Image.open(img_path)

    # Create a draw object
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = bbox

    # Draw rectangle (bbox)
    # Parameters: [(x1, y1), (x2, y2)], outline color, line width
    draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)

    # Display the image
    return img,text, bbox 
    # Corrected filter ranges for 1920x1080
    filter_ranges = {
        "top_left_corner": (0, 0, 640, 360),
        "top_right_corner": (1280, 0, 1920, 360),
        "bottom_left_corner": (0, 720, 640, 1080),
        "bottom_right_corner": (1280, 720, 1920, 1080),
        "top_middle_side": (640, 0, 1280, 360),
        "bottom_middle_side": (640, 720, 1280, 1080),
        "left_middle_side": (0, 360, 640, 720),
        "right_middle_side": (1280, 360, 1920, 720),
        "center_point": (640, 360, 1280, 720)
    }


filter_range= "top_middle_side"
# filter_range= "top_left_corner"
filter_range= "top_right_corner"
full_text = "AaBbc AaB AaBbCc AaBbccD"
object = 'heading 2'
object = 'titile'
object = 'A'
object = 'grid icon'
object = 'arrow down icon'
object = 'scrollbar'
object= "extension icon"
# object = "11"
# object = 'arrow up icon'
# object = 'heading'
# object = 'AaBb'
# object = 'A'
close = None
# close = 'add ins'
# close = 'find'
# close = '11'
# close = 'A'
# close = 'Calibri (Body)'
direction= 'right'
direction = None
# close = 'Heading 1'
n=1
img,text, bbox = get_points(filter_range,object,close,direction,n)
img





display(img)










import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import nest_asyncio
nest_asyncio.apply()


# def dubu_step(img_input,t,get_result,filter_range):
async def dubu_step(img_input, t, get_result, filter_range):
    global bbox_last
    async def do_step(imgg,t):
        global done,jitbbox,iimg,result_image_t,result_image,all_img,tbboxes,num,adding,addd,p,all_prompt,all_prompt_t,genai_output,genai_yesorno
        global bbox_last
        done = False
        bbox = None
        jitbbox = None
        num = None
        addd=None
        adding=None
        tbboxes = []
        all_img = []
        all_prompt=[]
        all_prompt_t = []
        p = """
Task: Analyze the content within the red bounding box (bbox) in the image.
Question: Does the red bounding box contain multiple (>1) elements/objects?
Required format: Respond with only one word - either "yes" or "no"
"""


        
        async def run_concurrent_tasks(result_image,prompt_t):
            start_time = time.time()
        
            async def task1():
                return await get_gen_async([result_image], prompt_t, "gemini-1.5-flash-exp-0827")
                # return await get_gen_async([result_image], prompt_t, "gemini-1.5-flash-002")
        
            async def task2():
                return await get_gen_async([result_image], p, "gemini-1.5-flash-002")
        
            # Run both tasks concurrently
            results = await asyncio.gather(task1(), task2())
        
            # Calculate total execution time
            end_time = time.time()
            total_time = end_time - start_time
        
            print(f"\nTotal concurrent execution time: {total_time:.2f} seconds")
        
            return results[0], results[1]
        
        # Assuming get_gen_async is an asynchronous version of get_gen
        async def get_gen_async(imgs, prompt, modeln):
            # Start time
            start_time = time.time()
            
            # Configure the API key (assuming rotator is defined elsewhere in your code)
            genai.configure(api_key=rotator.get_next_api_key())
            
            # Initialize the generative model
            model = genai.GenerativeModel(
                model_name=modeln,
                generation_config=generation_config,
            )
            
            # Prepare the input for the model
            inputs = [prompt] + imgs  # Concatenate prompt and list of images
            chat_session = model.start_chat(
                history=[]
            )
            
            # Use asyncio.to_thread to run the synchronous method in a separate thread
            response = await asyncio.to_thread(chat_session.send_message, inputs, stream=True)
            
            genai_output = ''
            
            # Stream the output
            for chunk in response:
                if chunk.text:
                    content = chunk.text
                    genai_output += content
                    print(content, end="", flush=True)
            
            print()  # New line after streaming is complete
            
            # End time
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            
            return genai_output
        
        async def main(result_image_t,prompt_t):
            global genai_output, genai_yesorno
        
            time.sleep(0.1)
            genai_output, genai_yesorno = await run_concurrent_tasks(result_image_t,prompt_t)
        
            if genai_output is None or genai_yesorno is None:
                logging.error("One or both tasks did not complete successfully.")
                # Handle the error appropriately, e.g., retry or exit the loop
                return


        def do_sam(img,conf):
            # Load and process image
            input_image = img
        
            # Run inference
            everything_results =modelfastsam(
                input_image,
                device=device,
                retina_masks=True,
                imgsz=1024,
                conf=conf,
                iou=0.9
            )
        
            # Process results
            prompt_process = FastSAMPrompt(input_image, everything_results, device=device)
            # Get bounding boxes
            bboxes = prompt_process.get_bboxes()
            return bboxes
        # def run_concurrent_tasks(result_image,prompt_t):
        #     # print(prompt_t)

        #     start_time = time.time()
        
        #     # Create a ThreadPoolExecutor to manage the threads
        #     with ThreadPoolExecutor(max_workers=2) as executor:
        #         # Submit both tasks
        #         future1 = executor.submit(get_gen, [result_image],prompt_t,"gemini-1.5-flash-002")
        #         future2 = executor.submit(get_gen, [result_image],p,"gemini-1.5-flash-002")
        #         # future1 = executor.submit(get_gen, [result_image],prompt_t,"gemini-1.5-flash-latest")
        #         # future2 = executor.submit(get_gen, [result_image],p,"gemini-1.5-flash-latest")
        
        #         # Get the results
        #         result1 = future1.result()
        #         result2 = future2.result()
        
        #     # Calculate total execution time
        #     end_time = time.time()
        #     total_time = end_time - start_time
        
        #     print(f"\nTotal concurrent execution time: {total_time:.2f} seconds")
        
        #     # # Print results
        #     # print("\nFirst function result:")
        #     # print(result1)
        #     # print("\nSecond function result:")
        #     # print(result2)
        #     return result1,result2
    
        def get_num(genai_output):
            # Extract the number using regex
            match = re.search(r'\d+', genai_output)
        
            if match:
                num = int(match.group())
                print(f"Extracted number: {num}")
            else:
                print("No number found in the text")
            return num
        def get_gen(imgs, prompt, modeln):
            # Start time
            start_time = time.time()
            
            # Configure the API key (assuming rotator is defined elsewhere in your code)
            genai.configure(api_key=rotator.get_next_api_key())
            
            # Initialize the generative model
            model= genai.GenerativeModel(
                model_name=modeln,
                generation_config=generation_config,
            )
            
            # Prepare the input for the model
            inputs = [prompt] + imgs  # Concatenate prompt and list of images
            chat_session = model.start_chat(
              history=[
              ]
            )
        
            response = chat_session.send_message( inputs , stream=True)
            
            genai_output = ''
            
            # Stream the output
            for chunk in response:
                if chunk.text:
                    content = chunk.text
                    genai_output += content
                    print(content, end="", flush=True)
            
            print()  # New line after streaming is complete
            
            # End time
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            print(f"\nProcessing time: {processing_time:.2f} seconds")
            
            return genai_output
        
    
    
        def is_too_close(bbox1, bbox2, min_distance=10):
            """
            Checks if two bounding boxes are too close based on a minimum distance threshold.
        
            Args:
                bbox1, bbox2: Bounding boxes in (x1, y1, x2, y2) format.
                min_distance: Minimum distance threshold.
        
            Returns:
                True if the bounding boxes are too close, False otherwise.
            """
            # Calculate the centers of the bounding boxes
            x1_center = (bbox1[0] + bbox1[2]) / 2
            y1_center = (bbox1[1] + bbox1[3]) / 2
            x2_center = (bbox2[0] + bbox2[2]) / 2
            y2_center = (bbox2[1] + bbox2[3]) / 2
        
            # Calculate Euclidean distance between the centers
            distance = np.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)
            return distance < min_distance
        
        def filter_close_bboxes(bboxes, min_distance=10):
            """
            Removes bounding boxes that are too close to each other.
        
            Args:
                bboxes: List of bounding boxes in (x1, y1, x2, y2) format.
                min_distance: Minimum distance threshold for filtering.
        
            Returns:
                Filtered list of bounding boxes.
            """
            filtered_bboxes = []
            for i, bbox in enumerate(bboxes):
                too_close = False
                for j, other_bbox in enumerate(filtered_bboxes):
                    if is_too_close(bbox, other_bbox, min_distance):
                        too_close = True
                        break
                if not too_close:
                    filtered_bboxes.append(bbox)
            return filtered_bboxes
        
        
        def display_adjusted_bboxes_on_image_cv2(input_image, bboxes, extracted_bbox, outline_color=(0, 0, 0), width=2, min_distance=20, scale_factor=4):
            global filtered_bboxes
            """
            Draws adjusted bounding boxes on the input image using OpenCV after filtering close boxes and scaling them.
        
            Args:
                input_image: PIL Image object.
                bboxes: List of bounding boxes, each in (x1, y1, x2, y2) format.
                extracted_bbox: The extracted bounding box (x_offset, y_offset).
                outline_color: Color for the bounding boxes (BGR format for OpenCV).
                width: Line width for drawing the boxes.
                min_distance: Minimum distance threshold for filtering close bounding boxes.
                scale_factor: Factor by which to scale the bounding boxes.
        
            Returns:
                PIL Image with filtered and scaled bounding boxes drawn.
            """
            # Convert the PIL image to OpenCV format (BGR)
            cv2_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
            # Filter out close bounding boxes
            filtered_bboxes = filter_close_bboxes(bboxes, min_distance)
        
            # Calculate the offset
            offset_x, offset_y = extracted_bbox[:2]
        
            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
        
            # Function to scale a bounding box
            def scale_bbox(bbox, factor):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Calculate the center of the bounding box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Scale the width and height
                new_width = width * factor
                new_height = height * factor
                
                # Calculate the new coordinates
                new_x1 = center_x - new_width / 2
                new_y1 = center_y - new_height / 2
                new_x2 = center_x + new_width / 2
                new_y2 = center_y + new_height / 2
                
                return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
            
            filtered = []
            for bbox in filtered_bboxes:
                    # Scale the bounding box
                    scaled_bbox = scale_bbox(bbox, scale_factor)
                    filtered.append(scaled_bbox)
            filtered_bboxes = filtered
            # Iterate over filtered bounding boxes, scale and draw them
            for idx, bbox in enumerate(filtered_bboxes):
        
                
                # Adjust the bounding box coordinates with the offset
                adjusted_bbox = (
                    int(bbox[0] + offset_x),
                    int(bbox[1] + offset_y),
                    int(bbox[2] + offset_x),
                    int(bbox[3] + offset_y)
                )
                
                # Draw the scaled and adjusted bounding box using OpenCV
                cv2.rectangle(
                    cv2_image,
                    (adjusted_bbox[0], adjusted_bbox[1]),
                    (adjusted_bbox[2], adjusted_bbox[3]),
                    outline_color,
                    width)
        
                # Prepare label text (index)
                label = str(idx)
                (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
                # Draw filled rectangle for label background
                cv2.rectangle(
                    cv2_image,
                    (adjusted_bbox[0], adjusted_bbox[1] - label_height - 5),
                    (adjusted_bbox[0] + label_width, adjusted_bbox[1]),
                    outline_color,
                    -1
                )
        
                # Put the label text
                cv2.putText(
                    cv2_image,
                    label,
                    (adjusted_bbox[0], adjusted_bbox[1] - 3),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    lineType=cv2.LINE_AA
                )
        
            # Convert the OpenCV image back to PIL format (RGB)
            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            
            return pil_image
        def optimized_draw_bboxes(image, bboxes, border_thickness=2):
            global filtered_bboxes
            """
            Optimized function to draw bounding boxes on images with better quality and performance.
            
            Args:
                image: PIL Image or numpy array
                bboxes: List of bounding boxes in format [x1, y1, x2, y2]
                border_thickness: Thickness of bbox borders
            
            Returns:
                PIL Image with drawn bounding boxes
            """
            print(f"Total bbox = {len(bboxes)}")
            start_time = time.time()
            
            # Ensure high quality image handling
            if isinstance(image, Image.Image):
                # Convert to high-quality numpy array
                image = np.array(image, dtype=np.uint8)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Create a copy to avoid modifying original
            image = image.copy()
            
            # Convert bboxes to numpy array for vectorized operations
            bboxes_array = np.array(bboxes, dtype=np.float32)
            
            # Define visually distinct colors
            colors = np.array([
                (0, 0, 0),
            ], dtype=np.uint8)
            
            def scale_bbox_vectorized(bboxes, scale_factor=1.2):
                """Vectorized implementation of bbox scaling"""
                center_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
                center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
                width = (bboxes[:, 2] - bboxes[:, 0]) * scale_factor
                height = (bboxes[:, 3] - bboxes[:, 1]) * scale_factor
                
                new_bboxes = np.column_stack([
                    center_x - width/2,
                    center_y - height/2,
                    center_x + width/2,
                    center_y + height/2
                ])
                return new_bboxes
            
            def check_overlaps_vectorized(bboxes):
                """Vectorized implementation of overlap checking"""
                n = len(bboxes)
                overlaps = np.zeros((n, n), dtype=bool)
                
                for i in range(n):
                    x1 = np.maximum(bboxes[i, 0], bboxes[:, 0])
                    y1 = np.maximum(bboxes[i, 1], bboxes[:, 1])
                    x2 = np.minimum(bboxes[i, 2], bboxes[:, 2])
                    y2 = np.minimum(bboxes[i, 3], bboxes[:, 3])
                    
                    overlap_width = np.maximum(0, x2 - x1)
                    overlap_height = np.maximum(0, y2 - y1)
                    overlaps[i] = (overlap_width * overlap_height) > 0
                    
                np.fill_diagonal(overlaps, False)
                return overlaps
            
            # Scale boxes and handle overlaps
            scaled_bboxes = scale_bbox_vectorized(bboxes_array)
            overlap_matrix = check_overlaps_vectorized(scaled_bboxes)
            
            final_bboxes = np.where(
                np.any(overlap_matrix, axis=1)[:, np.newaxis],
                bboxes_array,
                scaled_bboxes
            )
            
            # Filter small boxes
            areas = (final_bboxes[:, 2] - final_bboxes[:, 0]) * \
                    (final_bboxes[:, 3] - final_bboxes[:, 1])
            min_area = 100
            final_bboxes = final_bboxes[areas >= min_area]
            
        
            
            # Drawing parameters
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(image.shape[:2]) / 1000)  # Adaptive font scale
            thickness = max(1, min(image.shape[:2]) // 500)     # Adaptive thickness
            def remove_overlapping_boxes(boxes):
                """
                Remove all overlapping bounding boxes completely.
                Each box is in format [x1, y1, x2, y2].
                Input boxes can be numpy array or list.
                Returns a list of non-overlapping boxes and their indices.
                """
                # Convert to numpy array if not already
                boxes = np.array(boxes)
                
                if len(boxes) == 0:
                    return [], []
                    
                def boxes_overlap(box1, box2):
                    """Check if two boxes overlap at all"""
                    # If one rectangle is on left side of other
                    if box1[0] >= box2[2] or box2[0] >= box1[2]:
                        return False
                    
                    # If one rectangle is above other
                    if box1[1] >= box2[3] or box2[1] >= box1[3]:
                        return False
                        
                    return True
                
                # Calculate areas and sort indices
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                sorted_indices = np.argsort(-areas)  # Negative for descending order
                
                kept_boxes = []
                kept_indices = []
                
                # Check each box against all previously kept boxes
                for idx in sorted_indices:
                    current_box = boxes[idx]
                    overlaps = False
                    
                    for kept_box in kept_boxes:
                        if boxes_overlap(current_box, kept_box):
                            overlaps = True
                            break
                            
                    if not overlaps:
                        kept_boxes.append(current_box)
                        kept_indices.append(idx)
                
                return np.array(kept_boxes), np.array(kept_indices)
            filtered_bboxes, kept_indices = remove_overlapping_boxes(final_bboxes)
        
            # Sort boxes for consistent ordering
            sort_idx = np.lexsort((filtered_bboxes[:, 0],filtered_bboxes[:, 1]))
            filtered_bboxes = filtered_bboxes[sort_idx]
            
            # Draw boxes with anti-aliasing
            for idx, bbox in enumerate(filtered_bboxes):
                x1, y1, x2, y2 = map(int, bbox)
                color = colors[idx % len(colors)]
                
                # Draw anti-aliased rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), thickness, cv2.LINE_AA)
                
                # Prepare label
                label = f"{idx}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    image,
                    (x1, y1 - label_height - 5),
                    (x1 + label_width, y1),
                    color.tolist(),
                    -1,
                    cv2.LINE_AA
                )
                
                # Draw anti-aliased text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
            
            # Convert back to RGB with high quality
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            print(f"Processing time: {time.time() - start_time:.4f} seconds")
            print(f"Final BBoxes: {len(filtered_bboxes)}")
            
            return pil_image

        def interpolate_conf(img):
            width, height = img.size
            area = width * height
            print(f'area {area}')
    
            area1, conf1 = 230400, 0.33
            area2, conf2 = 3125, 0.5
    
            # Linear interpolation formula
            conf = conf1 + ((area - area1) / (area2 - area1)) * (conf2 - conf1)
            return conf
    
                    

            
        def last_bbox(input_image,bboxes):
            size = input_image.size
            start= [0, 0, size[0], size[1]]
            last = None
            for idx,i in enumerate(bboxes):
                print(idx)
                if last==None:
                    # Calculate the offset
                    offset_x = start[0]
                    offset_y = start[1]
                else:
                    idx = idx-1
                    # Calculate the offset
                    offset_x = last[0]
                    offset_y =last[1]
    
                # Adjust the bbox coordinates
                last= (
                    i[0] + offset_x,
                    i[1] + offset_y,
                    i[2] + offset_x,
                    i[3] + offset_y
                )
            print(last)
            return last
        def scale_bbox(bbox, scale_factor=1.1):
            """
            Scale a bounding box by a specified factor.
            Keeps the bbox centered while scaling.
            
            :param bbox: The original bounding box (x_min, y_min, x_max, y_max).
            :param scale_factor: The factor by which to scale the bounding box.
            :return: The scaled bounding box.
            """
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
        
            # Calculate the amount to expand on each side
            delta_width = (width * (scale_factor - 1)) / 2
            delta_height = (height * (scale_factor - 1)) / 2
        
            # Apply expansion
            x_min_new = max(0, int(x_min - delta_width))
            y_min_new = max(0, int(y_min - delta_height))
            x_max_new = int(x_max + delta_width)
            y_max_new = int(y_max + delta_height)
        
            return x_min_new, y_min_new, x_max_new, y_max_new
        def display_adjusted_bbox_on_image(input_image, bbox, outline_color=(0, 0, 255), width=2):
        
            las = last_bbox(input_image,tbboxes)
            # Assuming you have already opened the image:
            input = input_image.copy()
            las = scale_bbox(las, scale_factor=1.2)
            # Define bounding box coordinates
            x_min, y_min, x_max, y_max = las
            
            # Create a draw object
            draw = ImageDraw.Draw(input)
            
            # Draw the rectangle (outline)
            rectangle_color = (255, 0, 0)  # Red color
            rectangle_thickness = 2
            draw.rectangle([x_min, y_min, x_max, y_max], outline=rectangle_color, width=rectangle_thickness)
            
            # Display the image (optional)
            input = input.crop( bbox)
            
            return input
        inputimage = imgg.copy()
        img = imgg.copy()
        count = 0
    
    
    
        while done==False:
            count+=1
            print(count)
            if jitbbox is None:
    
                bbox = filter_ranges[filter_range]
                tbboxes.append(bbox)
                iimg = imgg
            else:
                bbox = jitbbox
                iimg = iimg.crop(bbox)
            
            
    
    
    
            s = time.time()
    
            # Convert PIL image to cv2 image
            image = cv2.cvtColor(np.array(iimg), cv2.COLOR_RGB2BGR)
    
            conf = interpolate_conf(iimg)
            print(f'confidence {conf}')
    
            bboxes = do_sam(iimg,conf)
            e = time.time()
            print(e-s)
            print(len(bboxes))
    
            if jitbbox is None:


                image = cv2.cvtColor(np.array(iimg), cv2.COLOR_RGB2BGR)
                result_image = optimized_draw_bboxes(image, bboxes, border_thickness=0.02)
                result_image.save('result_image.png')
                all_img.append(result_image)
            else:
                result_image = display_adjusted_bboxes_on_image_cv2(inputimage,bboxes,tbboxes[-1])
                result_image.save('result_image.png')
                all_img.append(result_image)


            more = f"rembember bbox id {num} isnot {object} !"

            # result_image = Image.open('result_image.png')
            prompt = f"""
# UI Element Detection and Bounding Box Assignment Task

## Objective
Identify and match specific UI elements with their corresponding bounding box IDs in user interface screenshots.

## Input Format
- **Image:** [UI screenshot with numbered bounding boxes]
- **Target UI Element:** [Detailed element specification]

## Required Output
- Single integer representing the correct bounding box ID
- Format: [ID]

## Element Specification Guidelines
1. **Location Details:**
   - Precise spatial position (top, bottom, left, right)
   - Relative positioning to other UI elements
   
2. **Visual Characteristics:**
   - Shape description
   - Icon/symbol details
   - Size information
   - Color properties (if relevant)
   
3. **Functional Description:**
   - Element type (button, icon, menu, etc.)
   - Purpose/functionality
4 . **Don't give any explaination**
   - think deeply
   - just the the output don't give any explaination
   
## Validation Rules
1. The selected bounding box must FULLY contain the target element
2. In case of overlapping boxes, select the most precise fit
3. Consider pixel-perfect alignment
4. Verify element boundaries match bounding box edges

## Example Query
"Identify the bounding box ID containing the extension icon, which appears as a small dark square symbol in the upper portion of the interface. The icon should be completely enclosed within the bounding box boundaries."

## Required Analysis Steps
1. Locate the target element using provided specifications
2. Identify ALL bounding boxes intersecting with the element
3. Verify complete containment
4. Select the most precise containing box
5. Double-check alignment and boundaries
6. Return only the box ID number

## Quality Checks
- Verify full element containment
- Confirm precise boundary alignment
- Validate against overlapping alternatives
- Ensure unique identification

## Output Format
Single integer only:
[Bounding Box ID]

-------------------
**Example:**

**Image:** [Image of a UI with numbered bounding boxes]
**Target Element:** "The dropdown menu located below the search bar"
**Output:** 17  (Assuming bounding box 17 corresponds to the described dropdown)


**Evaluation Criteria:**

The output will be evaluated based on accuracy. The correct bounding box ID must be returned for the given target element description.

-------------Your Targeted Job ------------
{more if addd else ''}
**Target Element:** {t}
**Output:**"""
            # print(prompt)
            all_prompt.append(prompt)
            
            
            genai_output = get_gen([result_image ],prompt,"gemini-1.5-flash-002")
            # genai_output = get_gen([result_image ],prompt,"gemini-1.5-flash-latest")
            
    
            
            
            num = get_num(genai_output)
            jitbbox = filtered_bboxes[num]
            tbboxes.append(jitbbox)
            
            
            # Assuming you have already opened the image:
            input_image = Image.open(img_path) 
    
            # result_image_t = display_adjusted_bbox_on_image(imgg,tbboxes[-1],tbboxes[-2])
    
            result_image_t =display_adjusted_bbox_on_image(input_image,tbboxes[0],tbboxes)
    
            all_img.append(result_image_t)
            

                    
            prompt_t = f"""
# UI Element Precise Detection Validator
A framework for validating exact UI element detection with bounding boxes

## Task Overview
Perform strict binary validation of object detection accuracy for specific UI elements, ensuring exact match between the targeted element and bounding box placement.

## Input Parameters
1. **Source Image**: UI screenshot or interface capture
2. **Target Element**: Precisely specified UI component to validate
   - Must include exact descriptor (e.g., "settings icon", "plus button", "menu icon")
   - Avoid ambiguous terms (e.g., just "icon" or "button")
3. **Detection Marker**: Red bounding box annotation

## Validation Protocol
### Primary Conditions
Return TRUE if and only if ALL conditions are met:
1. Red bounding box is present in the image
2. Bounding box encompasses EXACTLY the specified target element
   - Partial overlap is considered FALSE
   - Encompassing similar but different elements is FALSE
   - Encompassing the correct area but wrong element is FALSE

Return FALSE if ANY condition is met:
1. No red bounding box present
2. Bounding box encompasses wrong element
3. Bounding box encompasses similar element in correct area
4. Bounding box partially captures target element

### Important Distinctions
- Similar elements are NOT equivalent:
  - A plus icon ≠ extension icon
  - A menu button ≠ settings icon
  - A back arrow ≠ navigation icon
- Location context does not override element specificity:
  - Element being in correct area (e.g., extension area) does NOT validate incorrect element detection

## Output Specification
Required format:
```
prediction: <true/false>
```

## Validation Examples
Correct Evaluation:
```
Target: "plus icon"
Image: Contains red bbox around plus icon
Output: prediction: true
```

Incorrect Evaluation:
```
Target: "code icon"
Image: Contains red bbox around arrow down icon in code area
Output: prediction: false
Reason: Wrong element despite correct area
```
Correct Evaluation:
```
Target: "menu icon in left of dear"
Image: Contains a red bounding box around the menu icon in left of dear
Output: prediction: true
```

## Usage Instructions
1. Read target element specification carefully
2. Verify exact element match (not just similar elements)
3. Confirm precise bounding box placement
4. Provide binary output without explanation

## Key Reminders
- Exactness is paramount - similar is not equal
- Location context does not override element specificity
- When in doubt, verify target element descriptor matches exactly


## Your Task
Target: {t}
image: see the image i have provied to you!
"""
            # result_image_t.save('result_image.png')
            # result_image_t = Image.open('result_image.png')
            
            
            # genai_output = get_gen([result_image_t],prompt_t,"gemini-1.5-flash-002")
            await main(result_image_t, prompt_t)
            all_prompt_t.append(prompt_t)
            if 'false' in genai_output.lower() and 'no' in genai_yesorno.lower():
                tbboxes.pop()
                adding=True
                addd = True
                print('genai_output,genai_yesorno')
                more = f"rembember bbox id {num} isnot {object} !"
                prompt = f"""
# UI Element Detection and Bounding Box Assignment Task

## Objective
Identify and match specific UI elements with their corresponding bounding box IDs in user interface screenshots.

## Input Format
- **Image:** [UI screenshot with numbered bounding boxes]
- **Target UI Element:** [Detailed element specification]

## Required Output
- Single integer representing the correct bounding box ID
- Format: [ID]

## Element Specification Guidelines
1. **Location Details:**
   - Precise spatial position (top, bottom, left, right)
   - Relative positioning to other UI elements
   
2. **Visual Characteristics:**
   - Shape description
   - Icon/symbol details
   - Size information
   - Color properties (if relevant)
   
3. **Functional Description:**
   - Element type (button, icon, menu, etc.)
   - Purpose/functionality
4 . **Don't give any explaination**
   - think deeply
   - just the the output don't give any explaination
   
## Validation Rules
1. The selected bounding box must FULLY contain the target element
2. In case of overlapping boxes, select the most precise fit
3. Consider pixel-perfect alignment
4. Verify element boundaries match bounding box edges

## Example Query
"Identify the bounding box ID containing the extension icon, which appears as a small dark square symbol in the upper portion of the interface. The icon should be completely enclosed within the bounding box boundaries."

## Required Analysis Steps
1. Locate the target element using provided specifications
2. Identify ALL bounding boxes intersecting with the element
3. Verify complete containment
4. Select the most precise containing box
5. Double-check alignment and boundaries
6. Return only the box ID number

## Quality Checks
- Verify full element containment
- Confirm precise boundary alignment
- Validate against overlapping alternatives
- Ensure unique identification

## Output Format
Single integer only:
[Bounding Box ID]

-------------------
**Example:**

**Image:** [Image of a UI with numbered bounding boxes]
**Target Element:** "The dropdown menu located below the search bar"
**Output:** 17  (Assuming bounding box 17 corresponds to the described dropdown)


**Evaluation Criteria:**

The output will be evaluated based on accuracy. The correct bounding box ID must be returned for the given target element description.

-------------Your Targeted Job ------------
{more if addd else ''}
**Target Element:** {t}
**Output:**"""
                genai_output = get_gen([result_image ],prompt,"gemini-1.5-flash-002")
                # genai_output = get_gen([result_image ],prompt,"gemini-1.5-flash-latest")
                
        
                
                
                num = get_num(genai_output)
                jitbbox = filtered_bboxes[num]
                tbboxes.append(jitbbox)
                
                
                # Assuming you have already opened the image:
                input_image = Image.open(img_path) 
                def display_adjusted_bbox_on_image(input_image, bbox, outline_color=(0, 0, 255), width=2):
                
                    las = last_bbox(input_image,tbboxes)
                    # Assuming you have already opened the image:
                    input = input_image.copy()
                    las = scale_bbox(las, scale_factor=1.2)
                    # Define bounding box coordinates
                    x_min, y_min, x_max, y_max = las
                    
                    # Create a draw object
                    draw = ImageDraw.Draw(input)
                    
                    # Draw the rectangle (outline)
                    rectangle_color = (255, 0, 0)  # Red color
                    rectangle_thickness = 2
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=rectangle_color, width=rectangle_thickness)
                    
                    # Display the image (optional)
                    input = input.crop( bbox)
                    
                    return input
                result_image_t =display_adjusted_bbox_on_image(input_image,tbboxes[0],tbboxes)
        
                all_img.append(result_image_t)
                

                 
                
                # Run the main function
                await main(result_image_t, prompt_t)
                all_prompt_t.append(prompt_t)
                if 'false' in genai_output.lower() and 'no' in genai_yesorno.lower():
                    tbboxes.pop()
                    adding = True
                    addd = True
                    print('genai_output, genai_yesorno')
                    more = f"remember bbox id {num} is not {object} !"
                else:
                    break
                # result_image_t = display_adjusted_bbox_on_image(imgg,tbboxes[-1],tbboxes[-2])
        


                # result_image_t.save('result_image.png')
                # result_image_t = Image.open('result_image.png')
                
                # time.sleep(0.1)
                # genai_output = get_gen([result_image_t],prompt_t,"gemini-1.5-flash-002")
                # genai_output,genai_yesorno = run_concurrent_tasks(result_image_t)



                
            else:
                addd = False
                adding = False
            
            if 'true' in genai_output.lower():
                done=True
            if count>=2:
                break
    
        
        last = last_bbox(inputimage,tbboxes)
        
        
        # return last
        
    
    prompt_t = f"""
# UI Element Precise Detection Validator
A framework for validating exact UI element detection with bounding boxes

## Task Overview
Perform strict binary validation of object detection accuracy for specific UI elements, ensuring exact match between the targeted element and bounding box placement.

## Input Parameters
1. **Source Image**: UI screenshot or interface capture
2. **Target Element**: Precisely specified UI component to validate
   - Must include exact descriptor (e.g., "settings icon", "plus button", "menu icon")
   - Avoid ambiguous terms (e.g., just "icon" or "button")
3. **Detection Marker**: Red bounding box annotation

## Validation Protocol
### Primary Conditions
Return TRUE if and only if ALL conditions are met:
1. Red bounding box is present in the image
2. Bounding box encompasses EXACTLY the specified target element
   - Partial overlap is considered FALSE
   - Encompassing similar but different elements is FALSE
   - Encompassing the correct area but wrong element is FALSE

Return FALSE if ANY condition is met:
1. No red bounding box present
2. Bounding box encompasses wrong element
3. Bounding box encompasses similar element in correct area
4. Bounding box partially captures target element

### Important Distinctions
- Similar elements are NOT equivalent:
  - A plus icon ≠ extension icon
  - A menu button ≠ settings icon
  - A back arrow ≠ navigation icon
- Location context does not override element specificity:
  - Element being in correct area (e.g., extension area) does NOT validate incorrect element detection

## Output Specification
Required format:
```
prediction: <true/false>
```

## Validation Examples
Correct Evaluation:
```
Target: "plus icon"
Image: Contains red bbox around plus icon
Output: prediction: true
```

Incorrect Evaluation:
```
Target: "code icon"
Image: Contains red bbox around arrow down icon in code area
Output: prediction: false
Reason: Wrong element despite correct area
```
Correct Evaluation:
```
Target: "menu icon in left of dear"
Image: Contains a red bounding box around the menu icon in left of dear
Output: prediction: true
```

## Usage Instructions
1. Read target element specification carefully
2. Verify exact element match (not just similar elements)
3. Confirm precise bounding box placement
4. Provide binary output without explanation

## Key Reminders
- Exactness is paramount - similar is not equal
- Location context does not override element specificity
- When in doubt, verify target element descriptor matches exactly


## Your Task
Target: {t}
image: see the image i have provied to you!
"""

    genai_output = get_gen([img_input],prompt_t,"gemini-1.5-flash-exp-0827")
    # genai_output = get_gen([img_input],prompt_t,"gemini-1.5-flash-latest")
    if 'false' in genai_output.lower():
        print(True)
        
        img = Image.open(img_path)
        img = img.crop(filter_ranges[filter_range])
        await do_step(img,t)
        bbox = bbox_last
        x_min, y_min, x_max, y_max = bbox
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        text = f'x={x_center}, y={y_center}'
        print(text)

        # Assuming you have already opened the image:
        input_image = Image.open(img_path) 
        
        # Define bounding box coordinates
        x_min, y_min, x_max, y_max = bbox
        
        # Create a draw object
        draw = ImageDraw.Draw(input_image)
        
        # Draw the rectangle (outline)
        rectangle_color = (255, 0, 0)  # Red color
        rectangle_thickness = 2
        draw.rectangle([x_min, y_min, x_max, y_max], outline=rectangle_color, width=rectangle_thickness)
        
        # Display the image (optional)
        img = input_image
        return img, text, bbox
        
    else:
        
        return get_result
        
# pilimg,text, bbox = dubu_step(img,t,get_result,filter_range)
# genai_output = get_gen([img],prompt_t,"gemini-1.5-flash-002")

















from fs import *




# Set up parameters
model_path = "FastSAM-s.pt"
img_path = f"{home}/agi-.png"
# img_path = "/kaggle/input/dataaa/word.png"
# # img_path = "/kaggle/input/ddddddddddd/kaggle.png"
# img_path = f"{home}/wordd.png"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load model
modelfastsam = FastSAM(model_path)
# Load and process image
input_image = Image.open(img_path).convert("RGB")

def do_sam(img,conf):
    # Load and process image
    input_image = img

    # Run inference
    everything_results =modelfastsam(
        input_image,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=conf,
        iou=0.9
    )

    # Process results
    prompt_process = FastSAMPrompt(input_image, everything_results, device=device)
    # Get bounding boxes
    bboxes = prompt_process.get_bboxes()
    return bboxes
input_image = Image.open(img_path).convert("RGB")
bboxes = do_sam(input_image,0.2)




def do_sam(img,conf):
    # Load and process image
    input_image = img

    # Run inference
    everything_results =modelfastsam(
        input_image,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=conf,
        iou=0.9
    )

    # Process results
    prompt_process = FastSAMPrompt(input_image, everything_results, device=device)
    # Get bounding boxes
    bboxes = prompt_process.get_bboxes()
    return bboxes
s = time.time()
img_path = f"{home}/agi-.png"
img_path =  f"{home}/word.png"
input_image = Image.open(img_path).convert("RGB")
bboxes = do_sam(input_image,0.2)
e = time.time()
print(e-s)




perfect = []
for i, bbox in enumerate(bboxes):
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area>=2500 and area<551440:
            perfect.append(bbox)
            print(f"Bounding box {i+1} area: {area} pixels")
    else:
        print(f"Bounding box {i+1} is incomplete and cannot be calculated.")
len(perfect)




import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def optimized_draw_bboxes(image_path, bboxes, border_thickness=2):
    # Convert to numpy array for faster operations
    image = np.array(image_path)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Pre-define colors as numpy array for faster access
    bright_colors = np.array([
        (255, 165, 0),   # Orange
        (0, 0, 255),     # Blue
        (255, 0, 0),     # Red
        (0, 255, 0),     # Lime Green
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (50, 205, 50)    # Lime Green
    ], dtype=np.uint8)
    
    # Convert bboxes to numpy array for faster operations
    bboxes_array = np.array(bboxes)
    
    # Vectorized bbox filtering
    def is_bbox_inside_vectorized(bbox, all_bboxes):
        return np.any((all_bboxes[:, 0] <= bbox[0]) & 
                     (all_bboxes[:, 1] <= bbox[1]) & 
                     (all_bboxes[:, 2] >= bbox[2]) & 
                     (all_bboxes[:, 3] >= bbox[3]) & 
                     ~np.all(all_bboxes == bbox, axis=1))
    
    # Filter bboxes using vectorized operations
    filtered_indices = []
    for i, bbox in enumerate(bboxes_array):
        if not is_bbox_inside_vectorized(bbox, bboxes_array):
            filtered_indices.append(i)
    
    filtered_bboxes = bboxes_array[filtered_indices]
    
    # Sort bboxes - use numpy operations
    sort_indices = np.lexsort((filtered_bboxes[:, 0], filtered_bboxes[:, 1]))
    filtered_bboxes = filtered_bboxes[sort_indices]
    
    # Pre-calculate font parameters
    font_size = 30
    font = ImageFont.truetype("Arial.ttf", font_size)
    
    # Pre-calculate text dimensions for all numbers
    text_dimensions = {}
    for i in range(len(filtered_bboxes)):
        text = str(i)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_dimensions[i] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    # Draw boxes and text in batches
    for idx, bbox in enumerate(filtered_bboxes):
        color = tuple(bright_colors[idx % len(bright_colors)])
        
        # Draw rectangle
        draw.rectangle(bbox.tolist(), outline=color, width=8)
        
        # Calculate text position once
        text = str(idx)
        text_width, text_height = text_dimensions[idx]
        text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) // 2
        text_y = bbox[1] + (bbox[3] - bbox[1] - text_height) // 2
        
        # Draw text border in single operation if possible
        for dx in range(-border_thickness, border_thickness + 1, 2):
            for dy in range(-border_thickness, border_thickness + 1, 2):
                draw.text((text_x + dx, text_y + dy), text, font=font, fill=(0, 0, 0))
        
        # Draw colored text
        draw.text((text_x, text_y), text, font=font, fill=color)
    
    return pil_image

# Test the optimized version
print("Testing optimized PIL version...")
s = time.time()
image_with_filtered_bboxes = optimized_draw_bboxes(input_image, bboxes, border_thickness=7)
e = time.time()
print(f"Optimized PIL time: {e-s:.4f} seconds")


# Display the result
display(image_with_filtered_bboxes)






import cv2
import numpy as np
import time

def optimized_draw_bboxes(image, bboxes, border_thickness):
    print(f'total bbox = {len(bboxes)}')
    # Test the optimized version
    s = time.time()
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array
        image = np.array(image)
        # Convert RGB to BGR if the image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
    global filtered_bboxes
    # Convert bboxes to numpy array for vectorized operations
    bboxes_array = np.array(bboxes)
    
    # Pre-define colors as numpy array
    bright_colors = np.array([
        (255, 165, 0),   # Orange
        (0, 0, 255),     # Blue
        (255, 0, 0),     # Red
        (0, 255, 0),     # Lime Green
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (50, 205, 50)    # Lime Green
    ], dtype=np.uint8)
    
    # Vectorized bbox filtering
    def filter_bboxes_vectorized(bboxes_array):
        n_boxes = len(bboxes_array)
        x1 = bboxes_array[:, 0][:, np.newaxis]
        y1 = bboxes_array[:, 1][:, np.newaxis]
        x2 = bboxes_array[:, 2][:, np.newaxis]
        y2 = bboxes_array[:, 3][:, np.newaxis]
        
        inside_mask = (x1 >= bboxes_array[:, 0]) & \
                      (y1 >= bboxes_array[:, 1]) & \
                      (x2 <= bboxes_array[:, 2]) & \
                      (y2 <= bboxes_array[:, 3])
        
        np.fill_diagonal(inside_mask, False)
        return bboxes_array[~np.any(inside_mask, axis=1)]
    
    # Filter and sort bboxes
    filtered_bboxes = filter_bboxes_vectorized(bboxes_array)
    sort_idx = np.lexsort((filtered_bboxes[:, 0], filtered_bboxes[:, 1]))
    filtered_bboxes = filtered_bboxes[sort_idx]
    
    # Pre-calculate text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_sizes = np.array([cv2.getTextSize(str(i), font, font_scale, thickness)[0] 
                           for i in range(len(filtered_bboxes))])
    
    # Draw boxes in batches
    for idx, bbox in enumerate(filtered_bboxes):
        color = tuple(map(int, bright_colors[idx % len(bright_colors)]))
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=8)
        
        # Calculate text position once
        text = str(idx)
        text_width, text_height = text_sizes[idx]
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 + text_height) // 2
        
        # Draw black border using anti-aliasing with multiple shifts for a smoother edge
        for shift_x, shift_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cv2.putText(image, text, (text_x + shift_x, text_y + shift_y), font, font_scale, 
                        (0, 0, 0), thickness + border_thickness, lineType=cv2.LINE_AA)
        
        # Draw colored text on top with anti-aliasing
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    # Convert the color format from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image (NumPy array) to a PIL image
    pil_image = Image.fromarray(image_rgb)
    e = time.time()
    print(f"Optimized CV2 time: {e-s:.4f} seconds")

    return  pil_image


image = cv2.imread(img_path)  # Update with your image path

image_with_filtered_bboxes = optimized_draw_bboxes(image, bboxes, border_thickness=7)



display(image_with_filtered_bboxes)



def sk(bboxes):
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt


    # Calculate the width and height of each bounding box
    sizes = []
    for bbox in bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        sizes.append([width, height])

    # Convert sizes to a numpy array
    sizes = np.array(sizes)

    # Cluster the bounding boxes using KMeans
    num_clusters = 3  # Define the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sizes)

    # Print the cluster labels for each bounding box
    labels = kmeans.labels_
    print("Cluster labels for each bounding box:", labels)

    # Find the cluster with the majority of bounding boxes
    unique, counts = np.unique(labels, return_counts=True)
    majority_cluster = unique[np.argmax(counts)]
    print(f"Majority cluster: {majority_cluster}")

    # Print the bounding boxes in the majority cluster
    majority_bboxes = [bboxes[i] for i in range(len(bboxes)) if labels[i] == majority_cluster]
    print("Bounding boxes in the majority cluster:", majority_bboxes)
    return majority_bboxes











from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_bboxes_with_green_colors(image_path, bboxes, border_thickness=1, min_font_size=10, max_font_size=40):
    image = image_path.copy()
    draw = ImageDraw.Draw(image)

    # Calculate areas for all bounding boxes
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
    min_area, max_area = min(areas), max(areas)

    for idx, bbox in enumerate(bboxes):
        text = str(idx)
        bbox_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        text_color = (144, 238, 144)  # Light green color for text

        # Draw the bounding box
        draw.rectangle(bbox, outline= text_color, width=1)

        # Calculate font size based on bbox dimensions
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        font_size = int(np.interp(area, [min_area, max_area], [min_font_size, max_font_size]))

        # Load font with calculated size
        font = ImageFont.truetype("Arial.ttf", font_size)

        # Calculate the position for the text to be centered within the bbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2

        # Draw text with black border
        for offset_x in range(-border_thickness, border_thickness + 1):
            for offset_y in range(-border_thickness, border_thickness + 1):
                if offset_x != 0 or offset_y != 0:  # Skip the main text position
                    draw.text((text_x + offset_x, text_y + offset_y), text, font=font, fill=(0, 0, 0))  # Black border

        # Draw the main text
        draw.text((text_x, text_y), text, font=font, fill=text_color)

    return image


# image_with_filtered_bboxes = draw_bboxes_with_random_colors(img_path, bboxes, border_thickness=1, min_font_size=10, max_font_size=15)
# image_with_filtered_bboxes











class APIKeyRotator:
    def __init__(self, api_keys):
        self.api_keys = list(api_keys.values())
        self.current_index = 0
        self.request_count = 0

    def get_next_api_key(self):
        api_key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        self.request_count += 1
        return api_key

    def get_stats(self):
        return {
            "total_requests": self.request_count,
            "current_api_index": self.current_index
        }

# Usage example
api_keys = {
    "one":'AIzaSyAzxcRQRRF5sGUqm20nIzWtW4Jz1UJgMW4',
    "two":'AIzaSyDILr8QqALN7bo5724GndN7Lncp8rwOGc4',
    "three":"AIzaSyCSsDyfEJmcdnXfvwHy9l09O9eCMuQ3-3s",
    "four":"AIzaSyBipO-ucCEozl7xm_6EOmI1fbEmjjVRaio",
    "five":"AIzaSyChJOEy_n6cFDb9WsEQ78Qew8ivhVZBN88",
    "six":"AIzaSyDfACHw5xymLQM5wsLgIGQotFc4Lq2Skm8",
    "seven":"AIzaSyDTr4kgk7ylJt2cUp69auPK0NIJPbZVWu8",
    "eight":"AIzaSyCglqyXv2b2JlrOZ71SJp8HRQf7bBwC2wY",
    "nine":"AIzaSyCAI1Lo_7Oo2mEYjQv09mbPqO_0qLNddCI",
    "ten":"AIzaSyAdRLq06016E2rzNso1ZTvkgtzYqCXBJP8",
    "eleven":"AIzaSyBgvugAGBJ9Bojdrw8uRxKoKY0MKjk_pFQ",
    "twelve":"AIzaSyBOLjutFya6jfnJM81_hzTe4MkWyJHZ8-k",
}
rotator = APIKeyRotator(api_keys)








b = "right side black grid bellow :copy from input text black grid"
b ="blank document in the center_point"
prompt = f"""
As an AI you have to compelete (Your Targeted Job) by following according instructions and examples.
Your task is to identify the numbered box that corresponds to a specific section of the image. 


------------- Instruction ------------
Instructions:
1. Analyze the image carefully.
2. Locate the section described in the "Position" field.
3. Identify which numbered box covers the majority of that section.
4. Respond with ONLY the number of that box.

Important notes:
- Do not provide any explanations or additional text.
- The box numbers are color-coded to match the outline of the box they represent.
- Choose the box that covers the largest area of the specified section if multiple boxes overlap.


------------- Instruction ------------

------------- Example Job ------------
example1
position: Box under the left side of the draw!
num:11
example2
position: The center triangle section!
num:2
example3
position: Sign up section on the right!
num:24
------------- Example Job ------------


-------------Your Targeted Job ------------
position: {b}
num:
"""








import google.generativeai as genai
import time
# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}



from PIL import Image
import time
# modeln = "gemini-1.5-pro-exp-0801"
modeln = "gemini-1.5-pro-exp-0827"
# modeln = "gemini-1.5-flash"
# modeln = "gemini-1.5-pro-latest"
modeln = "gemini-1.5-flash-exp-0827"
modeln = "gemini-1.5-flash-002"
# modeln = "gemini-1.5-flash-latest"
# img = Image.open(img_path)

img = image_with_filtered_bboxes


def get_gen(imgs, prompt, modeln):
    # Start time
    start_time = time.time()
    
    # Configure the API key (assuming rotator is defined elsewhere in your code)
    genai.configure(api_key=rotator.get_next_api_key())
    
    # Initialize the generative model
    model= genai.GenerativeModel(
        model_name=modeln,
        generation_config=generation_config,
    )
    
    # Prepare the input for the model
    inputs = [prompt] + imgs  # Concatenate prompt and list of images
    chat_session = model.start_chat(
      history=[
      ]
    )

    response = chat_session.send_message( inputs , stream=True)
    
    genai_output = ''
    
    # Stream the output
    for chunk in response:
        if chunk.text:
            content = chunk.text
            genai_output += content
            print(content, end="", flush=True)
    
    print()  # New line after streaming is complete
    
    # End time
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    return genai_output
genai_output = get_gen([img],prompt,"gemini-1.5-flash-002")












import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small")














from pyngrok import ngrok

# Setting an auth token allows us to open multiple
# tunnels at the same time
ngrok.set_auth_token("2YGG5TlGcsNjutkY1hua39OM2gv_7XdWGdMHeh6KY4AqMjGiD")

# Start ngrok with the custom configuration file
ngrok_tunnel2 = ngrok.connect(8000, bind_tls=True)
print(ngrok_tunnel2)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import Response
import uuid
from PIL import Image
from pathlib import Path
import nest_asyncio
import aiofiles
import threading
from concurrent.futures import ThreadPoolExecutor

import asyncio

nest_asyncio.apply()
# global threshold, thread,img_path,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b 
global img_input
# Global variables to store the frame and image path
# global frame, img_path
frame = None
img_path = None
thread = None
app = FastAPI()
pp = []
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Input model for the request
class InputData(BaseModel):
    filter_range: str
    object: str
    close: str | None = None
    direction: str | None = None
    n: int
    verify: bool

class BoxInput(BaseModel):
    box: list[int]
def pro(file_path):
        frame = cv2.imread(file_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = position(frame)

@app.post("/specific_process/")
async def process_image(input_data: InputData):
    global thread,img_path,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b 
    global img_input
    def get_points(filter_range,object,close,direction,n,all_object,all_b):
        ai_output = f"""
        <filter_range>:{filter_range}
        <checked_object>: {object}
        <yestyping>:
        """
        if close!=None:
            ai_output = f"""
            <filter_range>:{filter_range}
            <checked_object>: {object}
            <close>:{close}
            <yestyping>:
            """
        action = "left click"
        text, bbox = return_bbox(ai_output,direction,filtered_results,action,n,all_object,all_b)
        
    
        # Load the image
        img = Image.open(img_path)
    
        # Create a draw object
        draw = ImageDraw.Draw(img)
    
        x1, y1, x2, y2 = bbox
    
        # Draw rectangle (bbox)
        # Parameters: [(x1, y1), (x2, y2)], outline color, line width
        draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=2)
    
        # Display the image
        return img,text, bbox 

    # Corrected filter ranges for 1920x1080
    filter_ranges = {
        "top_left_corner": (0, 0, 640, 360),
        "top_right_corner": (1280, 0, 1920, 360),
        "bottom_left_corner": (0, 720, 640, 1080),
        "bottom_right_corner": (1280, 720, 1920, 1080),
        "top_middle_side": (640, 0, 1280, 360),
        "bottom_middle_side": (640, 720, 1280, 1080),
        "left_middle_side": (0, 360, 640, 720),
        "right_middle_side": (1280, 360, 1920, 720),
        "center_point": (640, 360, 1280, 720)
    }

    # Validate input
    if input_data.close is None and input_data.direction is not None:
        raise HTTPException(
            status_code=400, 
            detail="If 'close' is None, 'direction' must also be None."
        )
    
    # Step 1: Call get_points
    try:
        get_result = get_points(
            input_data.filter_range, 
            input_data.object, 
            input_data.close, 
            input_data.direction, 
            input_data.n
            ,all_object,all_b
        )
        pilimg, text, bbox = get_result
        img_input = pilimg.crop(filter_ranges[input_data.filter_range])
        pp.append(pilimg)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error in get_points: {str(e)}"
        )
    
    # Step 2: Optional verification
    if input_data.verify:
        if input_data.close is None:
            t = f"""{input_data.object}"""
        else:
            t = f"""{input_data.object} {input_data.direction} of {input_data.close}"""
        try:
            # Await the async dubu_step function
            result = await dubu_step(img_input, f"{t}", get_result, input_data.filter_range)
            # if result is None or len(result) != 3:
            #     raise ValueError("Invalid result from dubu_step")
            pilimg, text, bbox = result
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    results = bbox
            
    # Step 3: Save image temporarily and send to user
    # file_path = f"/tmp/{uuid.uuid4()}.png"
    # pilimg.save(file_path)
    # return FileResponse(file_path, media_type="image/png", filename="processed_image.png")
    return JSONResponse(content={"results": results})


@app.post("/upload_process/")
async def upload_process(file: UploadFile = File(...), threshold: float = Form(...)):
    global thread,img_path,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b
    # Global variables to store the frame and image path
    global frame, img_path

    try:
        img_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(img_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                await buffer.write(chunk)

        # Read and process the image
        frame = cv2.imread(str(img_path))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Pass the frame and threshold to the position function
        results = position(frame, threshold=threshold)
        
        top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b = results
        
        # Return a success response
        # return {"message": "Image successfully processed.", "file_path": str(img_path), "threshold": threshold}
        return {"message": "Image successfully processed.", "file_path": str(img_path), "threshold": results}
    except Exception as e:
        return JSONResponse({"error": f"An error occurred: {str(e)}"}, status_code=500)
@app.post("/upload_only/")
async def upload_only(file: UploadFile = File(...), threshold: float = Form(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                await buffer.write(chunk)

        return {"message": "Image successfully processed.", "file_path": str(file_path), "threshold": threshold}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.post("/full_process/")
async def full():
    global thread,img_path,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b 

    try:

        results = top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b 

   
        # Return results as JSON
        return JSONResponse(content={"results": results})
 
        # return {"message": "Image successfully uploaded and saved.", "file_path": str(img_path)}
    except Exception as e:
        return JSONResponse({"error": f"An error occurred: {str(e)}"}, status_code=500)
@app.post("/fullimg/")
async def fullimg():
    global thread,img_path,top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b 

    start = time.time()
    
    
    img = process_image_with_bboxes(
        img_path,
        all_object,
        all_b,      
        output_path
    )
    
    end = time.time()
    print(f"Processing time: {end - start:.4f} seconds")
    
    # Convert the color format from BGR to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert the OpenCV image (NumPy array) to a PIL image
    pilimg = Image.fromarray(image_rgb)

    # Step 3: Save image temporarily and send to user
    file_path = f"/tmp/{uuid.uuid4()}.png"
    pilimg.save(file_path)
    return FileResponse(file_path, media_type="image/png", filename="processed_image.png")



@app.post("/predict/")
async def predict(box_input: BoxInput):
    global frame
    try:
        s = time.time()
        if frame is None:
            raise HTTPException(status_code=400, detail="No image frame available. Please upload an image first.")
        
        predictor.set_image(frame)
        
        input_box = np.array(box_input.box)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        e = time.time()
        print(f'total time = {e-s}')

        # Save to compressed npz format in memory
        output = io.BytesIO()
        np.savez_compressed(output, masks=masks, scores=scores)
        output.seek(0)

        return Response(content=output.getvalue(), media_type="application/octet-stream")
    except Exception as e:
        return JSONResponse({"error": f"An error occurred: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

