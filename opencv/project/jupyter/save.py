# import cv2
# import numpy as np

# drawing = False
# ix, iy = -1, -1
# thickness = 2  # Initial thickness

# def draw_mask(event, x, y, flags, param):
#     global ix, iy, drawing, img, mask, thickness
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing:
#             cv2.line(img, (ix, iy), (x, y), (0, 255, 0), thickness)
#             cv2.line(mask, (ix, iy), (x, y), 255, thickness)
#             ix, iy = x, y
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.line(img, (ix, iy), (x, y), (0, 255, 0), thickness)
#         cv2.line(mask, (ix, iy), (x, y), 255, thickness)

# def update_thickness(x):
#     global thickness
#     thickness = x

# def save_masked_image():
#     # Create a white image of the same size as the original
#     white_image = np.ones_like(original_img) * 255
    
#     # Use the mask to create the final image
#     final_image = cv2.bitwise_and(white_image, white_image, mask=mask)
    
#     cv2.imwrite('masked_image.png', final_image)
#     print("Masked image saved as 'masked_image.png'")

# # Load the image
# original_img = cv2.imread(r'dah.jpg')
# if original_img is None:
#     exit()

# img = original_img.copy()
# mask = np.zeros(img.shape[:2], np.uint8)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', img.shape[1], img.shape[0])
# cv2.setMouseCallback('image', draw_mask)

# # Create a trackbar for thickness
# cv2.createTrackbar('Thickness', 'image', thickness, 50, update_thickness)

# print("Press 's' to save the masked image, 'q' to quit")

# while True:
#     cv2.imshow('image', img)
#     cv2.imshow('mask', mask)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('s'):  # 's' key to save
#         save_masked_image()
#     elif k == ord('q'):  # 'q' key to exit
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
thickness = 2  # Initial thickness

def draw_mask(event, x, y, flags, param):
    global ix, iy, drawing, img, mask, thickness
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0, 255, 0), thickness)
            cv2.line(mask, (ix, iy), (x, y), 255, thickness)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 255, 0), thickness)
        cv2.line(mask, (ix, iy), (x, y), 255, thickness)

def update_thickness(x):
    global thickness
    thickness = x

def save_masked_image():
    # Create a white image of the same size as the original
    white_image = np.ones_like(original_img) * 255
    
    # Use the mask to create the final image
    final_image = cv2.bitwise_and(white_image, white_image, mask=mask)
    
    cv2.imwrite('masked_image.png', final_image)
    print("Masked image saved as 'masked_image.png'")

def resize_to_hd(image):
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate the new dimensions
    aspect_ratio = w / h
    if aspect_ratio > 1920 / 1080:
        new_w = 1920
        new_h = int(1920 / aspect_ratio)
    else:
        new_h = 1080
        new_w = int(1080 * aspect_ratio)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

# Load the image
original_img = cv2.imread(r'dah.jpg')
if original_img is None:
    exit()

# Resize the image if larger than HD
h, w = original_img.shape[:2]
if w > 1920 or h > 1080:
    original_img = resize_to_hd(original_img)
    cv2.imwrite('resized_image.png', original_img)
    print("Resized original image saved as 'resized_image.png'")

img = original_img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', img.shape[1], img.shape[0])
cv2.setMouseCallback('image', draw_mask)

# Create a trackbar for thickness
cv2.createTrackbar('Thickness', 'image', thickness, 50, update_thickness)

print("Press 's' to save the masked image, 'q' to quit")

while True:
    cv2.imshow('image', img)
    cv2.imshow('mask', mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):  # 's' key to save
        save_masked_image()
    elif k == ord('q'):  # 'q' key to exit
        break

cv2.destroyAllWindows()
