
# face_cut.py

import cv2
import os

def face_cut(img_path, save_path):
    img = cv2.imread(img_path)
    cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    # os.makedirs(save_path, exist_ok=True)
    facerect = cascade.detectMultiScale(img)
    for i, (x,y,w,h) in enumerate(facerect):
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        cv2.imwrite(save_path, face_img)

for l in range(1,13):
    for i in range(275):
        face_cut('capture_data/'+str(l)+'_'+str(i)+'.png', 'cut_data/'+str(l)+'_'+str(i)+'.png')

