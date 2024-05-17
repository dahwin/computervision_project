'''dahyun+darwin = dahwin'''
import time
import pyautogui
import subprocess
import  multiprocessing ,threading,_asyncio,concurrent
time.sleep(5)
x_c,y_c = pyautogui.position()
# print(x_c)
# print(y_c)
# x_c,y_c = 456,322
# print(pyautogui.position())
# pyautogui.click(x=1837, y=1013)
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


