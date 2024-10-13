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
    masked_img = cv2.bitwise_and(original_img, original_img, mask=mask)
    cv2.imwrite('masked_image.png', masked_img)
    print("Masked image saved as 'masked_image.png'")

# Load the image
original_img = cv2.imread('al.png')
if original_img is None:
    print("Error: Could not load image 'al.png'.")
    exit()

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