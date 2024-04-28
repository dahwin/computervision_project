"dahyun+darwin=dahwin"
import pyautogui
import time
import random
import string
time.sleep(5)
while True:
    # Sleep for 10 seconds before performing actions

    # Click on the specified position
    pyautogui.click(377, 310)

    # Generate a random string
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=40))  # Generates a random string of length 30

    # Type the generated random string
    pyautogui.typewrite(random_string)
    time.sleep(100)

