# -*- coding: utf-8 -*-
"""
Created on 2022-06-25

@author: Mufeng Geng
"""
import os
import torch.utils.data as data
from PIL import Image
import numpy as np

# Path of the data set
path_noisy = r"../data/train-noisy/"
path_clean = r"../data/train-clean/"

# Divide the data set according to the txt files
noisy_1_txt = r"../data/dataset_division/train_noisy_1.txt"
noisy_2_txt = r"../data/dataset_division/train_noisy_2.txt"
clean_txt = r"../data/dataset_division/train_clean.txt"

noisy_1_list = list()
noisy_2_list = list()
clean_list = list()
for line_noisy_1 in open(noisy_1_txt, "r"):
    line_noisy_1 = line_noisy_1[:-1]
    path_noisy_image_1 = os.path.join(path_noisy, line_noisy_1)
    noisy_1_list.append(path_noisy_image_1)

for line_noisy_2 in open(noisy_2_txt, "r"):
    line_noisy_2 = line_noisy_2[:-1]
    path_noisy_image_2 = os.path.join(path_noisy, line_noisy_2)
    noisy_2_list.append(path_noisy_image_2)

for line_clean in open(clean_txt, "r"):
    line_clean = line_clean[:-1]
    path_clean_image = os.path.join(path_clean, line_clean)
    clean_list.append(path_clean_image)

def get_Training_Set():
    return DatasetFromFolder(noisy_1_list, noisy_2_list, clean_list)

def load_image(filepath):
    image =Image.open(filepath)
    image = np.array(image).astype('float32')/255.0
    return np.expand_dims(image, axis=0)

class DatasetFromFolder(data.Dataset):
    def __init__(self, noisy_1_list, noisy_2_list,clean_list):
        super(DatasetFromFolder, self).__init__()
        self.noisy_1_list = noisy_1_list
        self.noisy_2_list = noisy_2_list
        self.clean_list = clean_list

    def __getitem__(self, index):
        noisy_1 = load_image(self.noisy_1_list[index])
        noisy_2 = load_image(self.noisy_2_list[index])
        clean = load_image(self.clean_list[index])
        return {"A": noisy_1, "B": noisy_2,  "C": clean}

    def __len__(self):
        return len(self.noisy_1_list)
