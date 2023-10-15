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
time.sleep(3)
def newclick(center_x, center_y):
    time.sleep(6)
    p.click(center_x,center_y)
    time.sleep(1)

def reddit():
    time.sleep(6)
    p.click(x=738, y=12)
    time.sleep(1.5)
    p.typewrite('https://www.reddit.com/')
    p.press('enter')
reddit()


def reddit_task(username):
    time.sleep(2)
    p.click(x=475, y=94)
    time.sleep(3)
    p.typewrite()
    p.typewrite(f'{username}')
    p.press('enter')
    time.sleep(2)
def raddit_subtask():
    p.click(x=1349, y=417)
    time.sleep(20)
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

    time.sleep(1)
    p.click(x=63, y=89)
    time.sleep(1)





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
    # cv2.imshow('Dawin', img_final)
    # captured_video.write(img_final)

    if int(time.time()) % 2 == 0:
        take_screenshot()
        count = 0
        # Split the results into lines
        lines = data.splitlines()

        # The first line contains the header, so we skip it
        # header = lines[0].split()
        lines = lines[1:]
        tab = ['reddit']
        while True:
            for T in tab:
                # Loop through each line and extract the bounding box coordinates for the words 'tab'
                for line in lines:
                    data = line.split()
                    detected_word = data[-1].lower()


                    if detected_word ==T:
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
                        print(center_x, center_y)
                        time.sleep(1)

                        for user in username:


                            if 'reddit' == T:
                                reddit_task(username=user)
                                if f"{user}" in detected_word:
                                    x, y, w, h = [int(x) for x in data[6:10]]
                                    if x >= 1125 and x <= 1450 and y >= 223 and y <= 950:
                                        center_x = x + w // 2
                                        center_y = y + h // 2
                                        newclick(center_x, center_y)
                                        raddit_subtask()



                        time.sleep(1)

                        break