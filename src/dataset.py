import os

import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image
import numpy as np

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class DataGenerator(Dataset):
    CHARS = '0123456789'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, path, sub_folder, images_list, labels, transform):
        self.path = path
        self.sub_folder = sub_folder
        self.images_list = images_list
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        grayimage = cv2.imread(
            self.path+"{0}{1}".format(self.sub_folder, self.images_list[idx]), 0)
        _, binary_image = cv2.threshold(
            grayimage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        opening = cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        binary_image = cv2.subtract(binary_image, opening)
        binary_image = (binary_image / 127.5) - 1.0
        img = torch.tensor(binary_image).unsqueeze(dim=0).float()
        img = self.transform(img)
        label = self.labels[idx]
        label = [self.CHAR2LABEL[c] for c in label]
        return img, torch.LongTensor(label), torch.LongTensor([len(label)])

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
