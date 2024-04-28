'dahyun+darwin=dahwin'

"dahyun+darwin=dahwin"
# 80% zoom out
import pyautogui
import time
# Wait for a moment to observe the scroll
time.sleep(10)


import random


while True:
    random_number = random.randint(20, 40)

    down = [pyautogui.scroll(-int(random_number)) for _ in range(10)]
    time.sleep(11)
    random_number = random.randint(20, 30)
    down = [pyautogui.scroll(int(random_number)) for _ in range(10)]