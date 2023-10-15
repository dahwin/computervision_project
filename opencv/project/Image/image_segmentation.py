'''dahyun+darwin = dahwin'''
import cv2

# Load the image
image = cv2.imread("D:\\dahyun\\Photobook\\1\Yes-I-am-Dahyun-Photobook-Scans-documents-15(2).jpeg")

# Resize the image to half its original size
resized_image = cv2.resize(image, (960, 540))

# Convert the image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert the image to binary
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Apply morphological operations to remove noise and fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)

# Apply connected component analysis to identify and label the individual objects in the image
output = cv2.connectedComponentsWithStats(closing, 4, cv2.CV_32S)

# Draw bounding boxes around the detected objects
for i in range(1, output[0]):
    x, y, w, h, area = output[2][i]
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the results
cv2.imshow('Input image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
