
import os
import pyautogui
import time

start = time.time()

for l in range(1,13):
    for i in range(275):
        im = pyautogui.screenshot('./capture_data/' + str(l) +'_'+ str(i) + '.png', region=(1050,50,800,450))
        time.sleep(5)

end = time.time()
print('result time is :', end - start)

