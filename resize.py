
# resize.py

import glob
import cv2
import os

names = ['other', 'ichika', 'nino', 'miku', 'yotsuba', 'itsuki']
for name in names:
    faces = glob.glob('data/'+name+'/*.png')
    for face in faces:
        f = cv2.imread(face, 1)
        resize_f = cv2.resize(f, (64,64))
        cv2.imwrite(face, resize_f)


