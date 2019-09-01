from torch.utils.data import Dataset as BaseDataset

import os
import cv2
import matplotlib.pyplot as plt


# Reference:
# https://github.com/alexgkendall/SegNet-Tutorial

class CamVidDataSet(BaseDataset):

    def __init__(self, train):

        self.CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                        'tree', 'signsymbol', 'fence', 'car',
                        'pedestrian', 'bicyclist', 'unlabelled']

        root_dir = "./src/data/CamVid/"
        classes = self.CLASSES

        if train:
            self.raw_images_dir = os.path.join(root_dir, 'train')
            self.masked_images_dir = os.path.join(root_dir, 'trainannot')
        else:
            self.raw_images_dir = os.path.join(root_dir, 'test')
            self.masked_images_dir = os.path.join(root_dir, 'testannot')

        self.ids = os.listdir(self.raw_images_dir)
        self.images_fps = [os.path.join(self.raw_images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(self.masked_images_dir, image_id) for image_id in self.ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        return image.transpose(2, 1, 0).astype('float'), mask.astype('float')


    def __len__(self):
        return len(self.ids)
