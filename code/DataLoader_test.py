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
path_noisy = r"../data/test-noisy/"

# Divide the data set according to the txt files
noisy_txt = r"../data/dataset_division/test_noisy.txt"

noisy_list = list()

for line_noisy in open(noisy_txt, "r"):
    line_noisy = line_noisy[:-1]
    path_noisy_image = os.path.join(path_noisy, line_noisy)
    noisy_list.append(path_noisy_image)

def get_Test_Set():
    return DatasetFromFolder(noisy_list)

def load_image(filepath):
    image =Image.open(filepath)
    image = np.array(image).astype('float32')/255.0
    return np.expand_dims(image, axis=0)

class DatasetFromFolder(data.Dataset):
    def __init__(self, noisy_list):
        super(DatasetFromFolder, self).__init__()
        self.noisy_list = noisy_list

    def __getitem__(self, index):
        noisy = load_image(self.noisy_list[index])
        return {"A": noisy}

    def __len__(self):
        return len(self.noisy_list)