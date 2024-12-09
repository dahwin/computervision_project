import json
from typing import List, Tuple, Dict
import numpy as np


        
def load_annotations_from_json(json_path: str) -> Tuple[List[str], List[List[float]]]:
    """
    Load object labels and bounding boxes from a JSON file
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Tuple containing:
        - List of object labels
        - List of bounding boxes
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = data['objects']
    bboxes = data['bboxes']
    
    return objects, bboxes

# Example usage:

# # Saving annotations
json_output_path = "annotations.json"
img_path= r"C:\Users\ALL USER\Desktop\e\ui\p.png"

# Loading annotations
loaded_objects, loaded_bboxes = load_annotations_from_json(json_output_path)

# Verify the data
print(f"\nNumber of objects loaded: {len(loaded_objects)}")
print(f"Number of bounding boxes loaded: {len(loaded_bboxes)}")

# Example of checking first few entries
print("\nFirst few entries:")
for obj, bbox in zip(loaded_objects[:3], loaded_bboxes[:3]):
    print(f"Object: {obj}")
    print(f"Bounding box: {bbox}\n")











import cv2
import numpy as np
import time
from PIL import Image

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


start = time.time()
output_path=None
img = process_image_with_bboxes(
    img_path,
   loaded_objects,
  loaded_bboxes ,      
    output_path
)

end = time.time()
print(f"Processing time: {end - start:.4f} seconds")

# Convert the color format from BGR to RGB
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (NumPy array) to a PIL image
pilimg = Image.fromarray(image_rgb)


pilimg.show()