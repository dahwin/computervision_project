import cv2
import pytesseract

# Load the image
image = cv2.imread('screenshot.png')
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"

# Perform OCR on the image
text = pytesseract.image_to_string(image)
print(text)
# Find the position of the text
boxes = pytesseract.image_to_data(image)
for i, b in enumerate(boxes.splitlines()):
    if i == 0:
        continue
    if 'DAHWIN' in b:
        x, y, w, h = (boxes.iloc[i, 6], boxes.iloc[i, 7], boxes.iloc[i, 8], boxes.iloc[i, 9])
        print("The coordinates of 'dahwin' text:", x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break

# Display the image with the text position
cv2.imshow("OCR result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
