
import os
import pickle
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform):
        self.transform = transform
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.data_dir = data['base_dir']
        self.image_list = data['list']
        
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.image_list[index][0])
        label = self.image_list[index][1]
        img = Image.open(img_path).convert('RGB')

        if self.transform!=None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)

    def get_weights(self):
        num = self.__len__()
        labels = np.zeros((num,), np.int)
        for s_ind, sample in enumerate(self.image_list):
            labels[s_ind] = sample[1]
        tmp = np.bincount(labels)
        weights = 1.0 / np.asarray(tmp[labels], np.float)

        return weights