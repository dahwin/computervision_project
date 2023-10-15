import pyautogui
import time

time.sleep(2)

for line in open("typing-data.txt", "r"):
    pyautogui.typewrite(line)

    pyautogui.press("enter")