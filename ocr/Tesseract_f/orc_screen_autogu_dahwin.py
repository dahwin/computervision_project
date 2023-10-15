'''dahyun+darwin = dahwin'''
import time
import pandas
import cv2
import pyautogui as p
import datetime
import time
from PIL import ImageGrab
import numpy as np
import cv2
from win32api import GetSystemMetrics
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract"
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
df = pandas.read_csv("C:\\Users\\Pc\\Desktop\\conputer_Vison\\opencv\\project\\video\\data\\fiverr\\chwasiullah\\result.csv")
username = df['username']
country = df['country-name']
# for countr in country:
count = 0
data = 0
time.sleep(3)
def newclick(center_x, center_y):
    time.sleep(6)
    p.click(center_x,center_y)
    time.sleep(1)

def facbook():

    time.sleep(6)
    p.click(x=346, y=1056)
    time.sleep(3)
    p.click(x=968, y=616)
    time.sleep(5)
    p.typewrite('https://www.facebook.com/')
    p.press('enter')

def instagram():
    time.sleep(6)
    p.click(x=258, y=13)
    time.sleep(1)
    p.typewrite('https://www.instagram.com/')
    p.press('enter')


def linkedin():
    time.sleep(6)
    p.click(x=502, y=8)
    time.sleep(1.5)
    p.typewrite('https://www.linkedin.com/')
    p.press('enter')


facbook()
instagram()
linkedin()
time.sleep(5)
def typing():
    p.typewrite("Hi There, I am MD Sofiullah from Fiverr. I see you take web scraping service from Fiverr. And, I have the most efficient and best gig on Fiverr on web scraping, data mining at the cheapest, reasonable price. You will get the highest quality service at a reasonable price.")
    time.sleep(2)
    p.press('enter')
    time.sleep(1)
    p.typewrite("I'm going to give you best offer: You can scrape data in this price range: ✅ 10k record's: 40 dollars ✅ 100k record's: 240 dollars ✅ 1 Million record's: 1500 dollars ")
    time.sleep(2)
    p.press('enter')
    time.sleep(1)
    p.typewrite("Please visit my gig link on Fiverr to learn more about this gig!")
    time.sleep(2)
    p.press('enter')
    time.sleep(1)
    p.typewrite("https://www.fiverr.com/share/Eo1X2K ")
    time.sleep(30)
    p.press('enter')
    time.sleep(1)
    p.typewrite('This is my another gig: In this gig you can scrape data up to 25k in 50 - 100 dollar price range. After that, if you want to do extra website scraping, you can do extra data extraction at 0.002 - 0.0035 rate! ✅ 50 dollar : Scrape records up to 25k from easy static website! 0.002 USD/row after 25k ✅ 85 dollar : Scrape records up to 25k from complex Dynamic website! 0.003 USD/row after 25k ✅ 100 dollar: Scrape records up to 25k from Captcha , Logging Required Website! 0.0035 USD/row after 25k')
    time.sleep(2)
    p.press('enter')
    time.sleep(1)
    p.typewrite('Link of my another gig: Please visit my gig link on Fiverr to learn more about this gig!')
    time.sleep(2)
    p.press('enter')
    time.sleep(1)
    p.typewrite('https://www.fiverr.com/share/oAkeKx ')
    time.sleep(24)
    p.press('enter')
    time.sleep(1)
    p.typewrite("Previously, I scraped 1000+ websites as a data miner in various companies! But just new in Fiverr. If I cannot deliver to your 100% satisfaction. Then I guarantee to refund you. So if you're looking for a skilled and reliable web scraper in reasonable price, look no further. I'm ready to help you achieve your goals and take your business to the next level. Feel free to Contact me today to discuss your project and see how I can help.")

def facebook_task(username):
    p.click(x=66, y=127)
    time.sleep(1)
    p.click(x=618, y=130)
    p.click(x=618, y=130)
    time.sleep(2)
    p.typewrite(f'{username}')
    p.press('enter')
    time.sleep(2)
    p.click(x=87, y=414)
    time.sleep(2)

def facebooksubtask():
    p.click(x=1597, y=319)
    time.sleep(3)
    typing()
    p.click(x=1796, y=574)
    time.sleep(1)
    p.click(x=21, y=48)
    time.sleep(1)
    p.click(x=33, y=132)

def instagram_task(username):
    p.click(x=44, y=150)
    time.sleep(2)
    p.click(x=62, y=279)
    time.sleep(2)
    p.typewrite(f"{username}")
def instagramsubtask():
    p.click(x=1298, y=153)
    time.sleep(3)
    typing()
    time.sleep(1)
    p.click(x=62, y=153)





def linkedin_task(username):
    time.sleep(2)
    p.click(x=495, y=129)
    time.sleep(2)
    p.typewrite(f'{username}')
    p.press('enter')
    time.sleep(2)
    p.click(x=422, y=185)
    time.sleep(2)





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
    # cv2.imshow('Dawin', img_final)
    # captured_video.write(img_final)

    if int(time.time()) % 2 == 0:
        take_screenshot()

        # Split the results into lines
        lines = data.splitlines()

        # The first line contains the header, so we skip it
        # header = lines[0].split()
        lines = lines[1:]
        tab = ['facebook','instagram', 'linkedin']
        while True:

            for user,y in zip(username[11:], country[11:]):
                    for T in tab:

                        if 'facebook' == T:
                            for line in lines:
                                data = line.split()
                                detected_word = data[-1].lower()

                                if detected_word == 'facebook':
                                    x, y, w, h = [int(x) for x in data[6:10]]
                                    print("Bounding box coordinates of the word:", detected_word)
                                    print("x:", x)
                                    print("y:", y)
                                    print("w:", w)
                                    print("h:", h)
                                    # Finding the center point
                                    center_x = x + w // 2
                                    center_y = y + h // 2
                                    newclick(center_x, center_y)
                                    time.sleep(2)
                                    facebook_task(username=user)
                                    time.sleep(4)
                                    for line in lines:
                                        data = line.split()
                                        detected_wordf = data[-1].lower()
                                        print(detected_wordf)
                                        try:
                                          if f"{user}" in detected_wordf:
                                              x, y, w, h = [int(x) for x in data[6:10]]
                                              if x >= 600 and x <= 1600 and y >= 41 and y <= 950:
                                                  center_x = x + w // 2
                                                  center_y = y + h // 2

                                                  newclick(center_x, center_y)

                                                  facebooksubtask()


                                        except:
                                         if f"{y}" in detected_wordf:

                                            x, y, w, h = [int(x) for x in data[6:10]]
                                            if x >= 600 and x <= 1600 and y >= 41 and y <= 950:
                                                center_x = x + w // 2
                                                center_y = y + h // 2

                                                newclick(center_x, center_y)

                                                facebooksubtask()

                        elif 'instagram' == T:
                            for line in lines:
                                data = line.split()
                                detected_word = data[-1].lower()

                                if detected_word == 'instagram':
                                    x, y, w, h = [int(x) for x in data[6:10]]
                                    print("Bounding box coordinates of the word:", detected_word)
                                    print("x:", x)
                                    print("y:", y)
                                    print("w:", w)
                                    print("h:", h)
                                    # Finding the center point
                                    center_x = x + w // 2
                                    center_y = y + h // 2
                                    newclick(center_x, center_y)
                                    time.sleep(2)
                                    time.sleep(6)
                                    instagram_task(username=user)
                                    time.sleep(4)
                                    for line in lines:
                                        data = line.split()
                                        detected_wordi = data[-1].lower()
                                        if f"{user}" in detected_wordi:
                                            x, y, w, h = [int(x) for x in data[6:10]]
                                            if x >= 80 and x <= 450 and y >= 225 and y <= 1035:
                                                center_x = x + w // 2
                                                center_y = y + h // 2
                                                newclick(center_x, center_y)
                                                if 'message' in detected_wordi:
                                                   instagramsubtask()
                        elif "fuck" == T:
                            pass
                        else:
                            pass


                        time.sleep(1)




                        # for user in username:
                        #     if f"{user}" in word.lower():
                        #         x, y, w, h = [int(x) for x in data[6:10]]
                        #         center_x = x + w // 2
                        #         center_y = y + h // 2
                        #         newclick(center_x, center_y)

                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break
