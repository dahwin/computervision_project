"dahyun+darwin=dahwin"
# 80% zoom out
import pyautogui
import time
# Wait for a moment to observe the scroll
time.sleep(10)

x_c,y_c = pyautogui.position()
# print(x_c)
# print(y_c)
# x_c,y_c = 456,322
# pyautogui.click(x_c,y_c)
# print(pyautogui.position())
# pyautogui.click(x=1837, y=1013)


while True:
    x = 0
    for _ in range(11):
        
        xx= 107*x
        if x ==0:
            print(True)
            xx = 0
        pyautogui.click(x_c+xx,y_c)
        time.sleep(0.5)
        pyautogui.click(x=1837, y=1013)
        time.sleep(0.5)
        x+=1
    down = [pyautogui.scroll(-int(14.5)) for _ in range(10)]
    time.sleep(1)