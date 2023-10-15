'''dahyun+darwin = dahwin'''
import time

# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
# img = cv2.imread("screen.png")
#
# # Detecting characters
# w1, h1, _ = img.shape
# results = pytesseract.image_to_data(img)
#
# # Split the results into lines
# lines = results.splitlines()
#
# # The first line contains the header, so we skip it
# # header = lines[0].split()
# lines = lines[1:]
# # Loop through each line and extract the bounding box coordinates for the word 'twice'
# for line in lines:
#     data = line.split()
#     word = data[-1]
#     # if word != "-1":
#     #     print(word)
#     # if word.lower() == "twice":
#     #     x, y, w, h = [int(x) for x in data[6:10]]
#     #     print("Bounding box coordinates of the word 'twice':")
#     #     print("x:", x)
#     #     print("y:", y)
#     #     print("w:", w)
#     #     print("h:", h)
#     #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     #     # Finding the center point
#     #     center_x = x + w // 2
#     #     center_y = y + h // 2
#     #     cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
#     #     print(center_x,center_y)
#
#     if "twice" in word.lower():
#         x, y, w, h = [int(x) for x in data[6:10]]
#         print("Bounding box coordinates of the word containing 'twice':")
#         print("x:", x)
#         print("y:", y)
#         print("w:", w)
#         print("h:", h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # Finding the center point
#         center_x = x + w // 2
#         center_y = y + h // 2
#         cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
#         print(center_x, center_y)
#
# cv2.imshow('Detected word "twice"', img)
# cv2.waitKey(0)
# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
# img = cv2.imread("screen.png")
#
# # Detecting characters
# w1, h1, _ = img.shape
# results = pytesseract.image_to_data(img)
#
# # Split the results into lines
# lines = results.splitlines()
#
# # The first line contains the header, so we skip it
# # header = lines[0].split()
# lines = lines[1:]
#
# found = False
#
# # Loop through each line and extract the bounding box coordinates for the word 'twice'
# for line in lines:
#     data = line.split()
#     word = data[-1]
#     if "twice" in word.lower() and not found:
#         x, y, w, h = [int(x) for x in data[6:10]]
#         print("Bounding box coordinates of the word containing 'twice':")
#         print("x:", x)
#         print("y:", y)
#         print("w:", w)
#         print("h:", h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # Finding the center point
#         center_x = x + w // 2
#         center_y = y + h // 2
#         cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
#         print(center_x, center_y)
#         found = True
#
# cv2.imshow('Detected word "twice"', img)
# cv2.waitKey(0)
# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
# img = cv2.imread("twice.png")
#
# # Detecting characters
# w1, h1, _ = img.shape
# results = pytesseract.image_to_data(img)
#
# # Split the results into lines
# lines = results.splitlines()
#
# # The first line contains the header, so we skip it
# # header = lines[0].split()
# lines = lines[1:]
#
# count = 0
#
# # Loop through each line and extract the bounding box coordinates for the word 'twice'
# for line in lines:
#     data = line.split()
#     word = data[-1]
#     if "twice" in word.lower():
#         x, y, w, h = [int(x) for x in data[6:10]]
#         print("Bounding box coordinates of the word containing 'twice':")
#         print("x:", x)
#         print("y:", y)
#         print("w:", w)
#         print("h:", h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # Finding the center point
#         center_x = x + w // 2
#         center_y = y + h // 2
#         cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
#         print(center_x, center_y)
#         count += 1
#         if count == 5:
#             break
#
# cv2.imshow('Detected word "twice"', img)
# cv2.waitKey(0)
# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
# img = cv2.imread("screen.png")
#
# # Detecting characters
# w1, h1, _ = img.shape
# results = pytesseract.image_to_data(img)
#
# # Split the results into lines
# lines = results.splitlines()
#
# # The first line contains the header, so we skip it
# # header = lines[0].split()
# lines = lines[1:]
#
# words_to_detect = ["twice", "release",'target']
#
# while True:
#     for word in words_to_detect:
#         for line in lines:
#             data = line.split()
#             detected_word = data[-1].lower()
#             if detected_word == word:
#                 x, y, w, h = [int(x) for x in data[6:10]]
#                 print("Bounding box coordinates of the word:", detected_word)
#                 print("x:", x)
#                 print("y:", y)
#                 print("w:", w)
#                 print("h:", h)
#                 # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 # Finding the center point
#                 center_x = x + w // 2
#                 center_y = y + h // 2
#                 # cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
#                 print(center_x, center_y)
#
#                 time.sleep(2)
#
#                 break



# cv2.imshow('Detected words', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
# img = cv2.imread("twice.png")
#
# # Define the rectangle in your image
# top_left = (600, 41)
# top_right = (1600, 41)
# bottom_left = (600, 950)
# bottom_right = (1600, 950)
# cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
#
# # Get the OCR results within the defined rectangle
# rect = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
# results = pytesseract.image_to_data(rect)
#
# # Split the results into lines
# lines = results.splitlines()
#
# # The first line contains the header, so we skip it
# # header = lines[0].split()
# lines = lines[1:]
#
# count = 0
#
#
# # Loop through each line and extract the bounding box coordinates for the word 'twice'
# for line in lines:
#     data = line.split()
#     word = data[-1]
#     if "twice" in word.lower():
#         x, y, w, h = [int(x) for x in data[6:10]]
#         print("Bounding box coordinates of the word containing 'twice':")
#         print("x:", x)
#         print("y:", y)
#         print("w:", w)
#         print("h:", h)
#         cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # Finding the center point
#         center_x = x + w // 2
#         center_y = y + h // 2
#         cv2.circle(rect, (center_x, center_y), 5, (0, 0, 255), -1)
#         print(center_x, center_y)
#         count += 1
#         if count == 5:
#             break
#
# cv2.imshow('Detected word "twice"', rect)
# cv2.waitKey(0)
import pyautogui as p

def newclick(center_x, center_y):
    time.sleep(6)
    p.click(center_x,center_y)
    time.sleep(1)
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
img = cv2.imread("twice.png")

# Detecting characters
w1, h1, _ = img.shape
results = pytesseract.image_to_data(img)

# Split the results into lines
lines = results.splitlines()

# The first line contains the header, so we skip it
# header = lines[0].split()
lines = lines[1:]
# # Define the rectangle in your image
# top_left = (600, 41)
# top_right = (1600, 41)
# bottom_left = (600, 950)
# bottom_right = (1600, 950)
count = 0
for line in lines:
    data = line.split()
    word = data[-1]
    if "twice" in word.lower():
        x, y, w, h = [int(x) for x in data[6:10]]
        if x >= 600 and x <= 1600 and y >= 41 and y <= 950:
            # print("Bounding box coordinates of the word containing 'twice':")
            # print("x:", x)
            # print("y:", y)
            # print("w:", w)
            # print("h:", h)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Finding the center point
            center_x = x + w // 2
            center_y = y + h // 2
            # cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
            j = (center_x, center_y)

            time.sleep(3)
            print(j)

            count += 1
            if count ==2 :
                break

            print('hi')

# cv2.imshow('Detected word "twice"', img)
# cv2.waitKey(0)