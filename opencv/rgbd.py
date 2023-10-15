import cv2

image_path = r"D:\dahyun\Photobook\photo\images.jpg"
bgr_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

cv2.imshow("RGB Image", rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
