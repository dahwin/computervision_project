import pyautogui
import time
time.sleep(2)
buttonx, buttony = pyautogui.locateCenterOnScreen('research.txt') # returns (x, y) of matching region
pyautogui.click(buttonx, buttony)  # clicks the center of where the button was found