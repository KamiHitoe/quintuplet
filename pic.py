

import os
import pyautogui
import time

start = time.time()
# ep_num = input('input ep num :')

img = pyautogui.screenshot('./capture_data/0_10.png', region=(1050,50,800,450))


end = time.time()
# print('result time is :', end - start)


