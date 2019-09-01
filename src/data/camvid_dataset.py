from torch.utils.data import Dataset as BaseDataset

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Reference:
# https://github.com/alexgkendall/SegNet-Tutorial

class CamVidDataSet(BaseDataset):

    def __init__(self, train, classes=None):

        self.CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
                        'tree', 'signsymbol', 'fence', 'car',
                        'pedestrian', 'bicyclist', 'unlabelled']

        print(os.getcwd())
        root_dir = "./src/data/CamVid/"
        #root_dir = "./CamVid"
        classes = self.CLASSES
        #self.ids = os.listdir(root_dir)

        if train:
            self.raw_images_dir = os.path.join(root_dir, 'train')
            self.masked_images_dir = os.path.join(root_dir, 'trainannot')
        else:
            self.raw_images_dir = os.path.join(root_dir, 'test')
            self.masked_images_dir = os.path.join(root_dir, 'testannot')

        self.ids = os.listdir(self.raw_images_dir)
        self.images_fps = [os.path.join(self.raw_images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(self.masked_images_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], 0)
        # print(mask.shape)
        # print(mask.shape) -> (360, 480)

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # print(type(masks)) -> <class 'list'>
        # print(masks)
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(mask)
        # print(mask.shape) #-> (360, 480, 12)

        # return image.transpose(2, 1, 0), mask
        return image.transpose(2, 1, 0).astype('float'), mask.astype('float')


    def __len__(self):
        return len(self.ids)


def visualize(**images):
    n = len(images)
    # print("num of images...%s" % n)
    plt.figure(figsize=(16, 5))
    for i, (key, value) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(key.split('_')).title())
        plt.imshow(value)
    plt.show()


# dataset = CamVidDataSet(train=True)
#
# image, mask = dataset[1]
#
# # print(image.shape) -> (360, 480, 3)
# # print(mask.shape)# -> (360, 480, 1)
# # print(type(image)) ->  <class 'numpy.ndarray'>
# # print(type(mask)) ->  <class 'numpy.ndarray'>
#
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )
