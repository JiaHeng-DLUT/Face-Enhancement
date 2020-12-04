#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import face_model
import argparse
import cv2
import glob
import sys
import numpy as np
import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
start = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

def get_embedding(img_path):
    img_cv2 = cv2.imread(img_path)
    img = model.get_input(img_cv2)
    if img is None:
        return None
    return model.get_feature(img)

def predict_id(f1):
    min_dist = 1e9
    id_predicted = 0
    for i in range(len(template_embedding_list)):
        f2 = template_embedding_list[i]
        dist = np.sum(np.square(f1 - f2))
        if (dist < min_dist):
            min_dist = dist
            id_predicted = template_id_list[i]
    return (id_predicted, min_dist)

template_embedding_list = []
template_id_list = []

base_dir = '/home/jiaheng/Desktop/GitHub/IJB-C-1/GT'
f = open('hr.txt', 'r')
out = f.readlines()
for line in out:
    _ = line.split(' ')
    # print(_)
    id = _[0]
    _ = _[1].split('/')
    filename = _[-1]
    # print(filename)
    filename = filename.replace('txt', 'jpg').rstrip('\n')
    img_path = os.path.join(base_dir, str(id)) + '/img' + '/' + filename
    print(img_path)
    # img_cv2 = cv2.imread(img_path)
    # print(img_cv2.shape)
    # cv2.imshow('', img_cv2)
    # cv2.waitKey(0)
    f2 = get_embedding(img_path)
    if f2 is not None:
        template_embedding_list.append(f2)
        template_id_list.append(id)
    print('Num of template embeddings:', len(template_embedding_list))
    #break
f.close()


# In[3]:


import glob
img_path_list = glob.glob('/home/jiaheng/Desktop/GitHub/IJB-C-1/LR_804_in_one_dir_enhanced_OPR/final_output/*.png')
n = len(img_path_list)
print(n)


# In[4]:

import pandas as pd
df = pd.read_csv('lr_80.4.txt', sep=' ', header=None)
df = df.rename({0:"idx", 1:"video_id", 2:"filename"}, axis="columns")
print(df.head(3))

import matplotlib.pyplot as plt
import random
fig = plt.figure(figsize=(8, 8))
rows = 3 
columns = 3
base_dir = '/home/jiaheng/Desktop/GitHub/IJB-C-1'
for i in range(rows):
    index = int(random.random() * n)
    s = df.iloc[index]
    print(s)
    idx = s['idx']
    video_id = s['video_id']
    filename = s['filename']
    OPR_path = base_dir + '/LR_804_in_one_dir_enhanced_OPR/final_output/' + filename[:-3] + 'png'
    LR_path = base_dir + '/LR_80.4/' + str(idx) + '/' + str(video_id) + '/' + str(filename)
    UMSN_path = base_dir + '/LR_deblurred_UMSN/' + str(idx) + '/' + str(video_id) + '/' + str(filename)
    print(OPR_path)
    print(LR_path)
    print(UMSN_path)
    fig.add_subplot(rows, columns, i * 3 + 1)
    f = get_embedding(LR_path)
    if f is not None:
        (id_p, min_d) = predict_id(f)
    else:
        (id_p, min_d) = (-1, -1)
    plt.title('LR\nID: {}\nPredict: {}\n Min dist: {}'.format(idx, id_p, min_d), fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(cv2.imread(LR_path), cv2.COLOR_BGR2RGB))
    plt.tight_layout()

    fig.add_subplot(rows, columns, i * 3 + 2)
    f = get_embedding(UMSN_path)
    if f is not None:
        (id_p, min_d) = predict_id(f)
    else:
        (id_p, min_d) = (-1, -1)
    plt.title('LR deblured by UMSN\nID: {}\nPredict: {}\n Min dist: {}'.format(idx, id_p, min_d), fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(cv2.imread(UMSN_path), cv2.COLOR_BGR2RGB))

    fig.add_subplot(rows, columns, i * 3 + 3)
    f = get_embedding(OPR_path)
    if f is not None:
        (id_p, min_d) = predict_id(f)
    else:
        (id_p, min_d) = (-1, -1)
    plt.title('LR restored by Old-Photo-Restoration\nID: {}\nPredict: {}\n Min dist: {}'.format(idx, id_p, min_d), fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(cv2.imread(OPR_path), cv2.COLOR_BGR2RGB))
plt.savefig('visualization_LR-UMSN-OPR.png')
plt.show()


# In[ ]:




