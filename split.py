
# split.py

import numpy as np
import glob
import cv2
from keras.utils.np_utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

names = ['other', 'ichika', 'nino', 'miku', 'yotsuba', 'itsuki']
img_list = []
label_list = []

# append index
for index, name in enumerate(names):
    face_img = glob.glob('data/'+name+'/*.png')
    for face in face_img:
        # imread RGB
        a = cv2.imread(face, 1)
        b = np.expand_dims(a, axis=0)
        img_list.append(b)
        label_list.append(index)

# convert pandas
X_pd = pd.Series(img_list)
y_pd = pd.Series(label_list)

# merge
Xy_pd = pd.concat([X_pd, y_pd], axis=1)
# shuffle
sf_Xy = Xy_pd.sample(frac=1)
# shuffle後にlistとして再取得
img_list = sf_Xy[0].values
label_list = sf_Xy[1].values
# tuple化して結合
X = np.r_[tuple(img_list)]
# convert binary
Y = to_categorical(label_list)

train_rate = 0.8

train_n = int(len(X) * train_rate)
train_X = X[:train_n]
test_X = X[train_n:]
train_y = Y[:train_n][:]
test_y = Y[train_n:][:]

## define scratch_functions

# 左右反転
def flip(img):
    flip_img = cv2.flip(img, 1)
    return flip_img

# ぼかし
def blur(img):
    blur_img = cv2.GaussianBlur(img, (5,5), 0)
    return blur_img

# γ変換
def gamma(img):
    gamma = 0.75
    LUT_G = np.arange(256, dtype = 'uint8')
    for i in range(256):
        LUT_G[i] = 255 * pow(float(i) / 255, 1.0 / gamma)
    gamma_img = cv2.LUT(img, LUT_G)
    return gamma_img

total_img = []
for x in train_X:
    imgs = [x]
    # concat list
    imgs.extend(list(map(flip, imgs)))
    imgs.extend(list(map(blur, imgs)))
    imgs.extend(list(map(gamma, imgs)))
    total_img.extend(imgs)

# add dims to total_img
img_expand = list(map(lambda x:np.expand_dims(x, axis=0), total_img))
# tuple化して結合
train_X_scratch = np.r_[tuple(img_expand)]

labels = []
for label in range(len(train_y)):
    lbl = []
    for i in range(2**3):
        lbl.append(train_y[label, :])
    labels.extend(lbl)

label_expand = list(map(lambda x:np.expand_dims(x, axis=0), labels))
train_y_scratch = np.r_[tuple(label_expand)]



