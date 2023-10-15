'''dahyun+darwin = dahwin'''
# import time
# while True:
#     d = 'https://www.google.com/search?q=how+to+delete+speac+in+pycharm&rlz=1C1CHBD_redditnBD1041BD1041&oq=how+to&aqs=chrome.1.69i57j69i59l3j0i433i512j69i60l2j69i61.9071facebookj0j7&sourceid=chrome&ie=UTF-8'
#
#     tab = ['google', 'facebook', 'twitter', 'instagram', 'linkedin', 'reddit']
#     for T in tab:
#
#         if f'{T}' in d.lower():
#             print(True)
#
#         else:
#             print(False)
#         time.sleep(4)
# import time
# def facebook():
#     bb = 5+4
#     time.sleep(2)
# def instagram():
#     bb = 5+4
#     time.sleep(2)
# def linledin():
#     bb = 5+4
#     time.sleep(2)
# def reddit():
#     bb = 5+4
#     time.sleep(2)
# while True:
#     d = 'https://www.google.com/search?q=how+to+delete+speac+in+pycharm&r.linkedin.lz=1C1CHBD_.reddit.nBD1041BD1041&oq=how+to&aqs=chrome.1.69i57j69i59l3j0i433i512j69i60l2j69i61.9071facebookj0j7&sourceid=chrome&ie=UTF-8'
#
#     tab = ['facebook','instagram', 'linkedin', 'reddit']
#     for T in tab:
#
#         if T in d.lower():
#             facebook()
#             print('facebook')
#
#         elif T in  d.lower():
#             instagram()
#             print('instagram')
#         elif T in  d.lower():
#             linledin()
#             print('linkedin')
#         elif T in  d.lower():
#             reddit()
#             print('reddit')

# import time
#
def facebook(U):
    bb = U + 'Love'
    print(bb)
    time.sleep(2)
    print('facebooktask')

def instagram(U):
    bb = U + 'Love'
    print(bb)
    time.sleep(2)
    print('instagramtask')

def linkedin(U):
    bb = U + 'Love'
    print(bb)
    time.sleep(2)
    print('linkedin task')

def reddit(U):
    bb = U+'Love'
    print(bb)
    time.sleep(2)
    print('reddit task')
#
# do = ['dahwin','dahyun','bubu','tofu']
# d = 'https://www.google.com/search?q=how+to+delete+speac+in+pycharm&r.linkedin.lz=1C1CHBD_.reddit.nBD1041BD1041&oq=how+to&aqs=chrome.1.69i57j69i59l3j0i433i512j69i60l2j69i61.9071facebookj0j7&sourceid=chrome&ie=UTF-8'
#
# while True:
#     for u in do:
#         for T in ['facebook','instagram','linkedin','reddit']:
#             if T in d.lower():
#                 if T == 'facebook':
#                     facebook(u)
#                 elif T == 'instagram':
#                     instagram(u)
#                 elif T == 'linkedin':
#                     linkedin(u)
#                 elif T == 'reddit':
#                     reddit(u)
#                 time.sleep(1) # Wait for 1 second between each function call


# import time
#
# def facebook():
#     bb = 5+4
#     time.sleep(2)
#
# def instagram():
#     bb = 5+4
#     time.sleep(2)
#
# def linkedin():
#     bb = 5+4
#     time.sleep(2)
#
# def reddit():
#     bb = 5+4
#     time.sleep(2)
#
# tab = ['facebook', 'instagram', 'linkedin', 'reddit']
# index = 0
#
# while True:
#     T = tab[index]
#     d = 'https://www.google.com/search?q=how+to+delete+speac+in+pycharm&r.linkedin.lz=1C1CHBD_.reddit.nBD1041BD1041&oq=how+to&aqs=chrome.1.69i57j69i59l3j0i433i512j69i60l2j69i61.9071facebookj0j7&sourceid=chrome&ie=UTF-8instagram'
#
#     if T in d.lower():
#         if T == 'facebook':
#             facebook()
#         elif T == 'instagram':
#             instagram()
#         elif T == 'linkedin':
#             linkedin()
#         elif T == 'reddit':
#             reddit()
#
#         print(T)
#         time.sleep(1)
#
#         index = (index + 1) % len(tab) # move to the next function in the list
#     else:
#         index = (index + 1) % len(tab) # move to the next function in the list

do = ['dahwin','dahyun','bubu','tofu']
d = 'https://www.google.com/search?q=how+to+delete+speac+in+pycharm&r.linkedin.lz=1C1CHBD_.reddit.nBD1041BD1041&oq=how+to&aqs=chrome.1.69i57j69i59l3j0i433i512j69i60l2j69i61.9071facebookj0j7&sourceid=chrome&ie=UTF-8'
import time
from PIL import ImageGrab
import numpy as np
import cv2
from win32api import GetSystemMetrics
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

count = 0
data = 0

def take_screenshot():
    img = ImageGrab.grab()
    img_np = np.array(img)
    img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    global data
    data = pytesseract.image_to_data(img_final)


while True:

    img = ImageGrab.grab()

    img_np = np.array(img)
    img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('Dawin', img_final)
    # captured_video.write(img_final)

    if int(time.time()) % 2 == 0:
        take_screenshot()
        print(data)
        # Split the results into lines
        lines = data.splitlines()

        for line in lines:
            data = line.split()
            detected_word = data[-1].lower()
            print(detected_word)
            while True:
                for u in do:
                    for T in ['facebook','instagram','linkedin','reddit']:
                        if T in d.lower():
                            if T == 'facebook':
                                facebook(u)

                            elif T == 'instagram':
                                instagram(u)
                            elif T == 'linkedin':
                                linkedin(u)
                            elif T == 'reddit':
                                reddit(u)
                            time.sleep(1) # Wait for 1 second between each function call


