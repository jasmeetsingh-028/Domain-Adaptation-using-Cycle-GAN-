from cv2 import transform
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class hzDataset(Dataset):
    def __init__(self, root_dir_zebra, root_dir_horse, transform= None):

        self.root_dir_zebra = root_dir_zebra
        self.root_dir_horse = root_dir_horse
        self.transform = transform

        self.horse_Imagename = os.listdir(root_dir_horse)
        self.zebra_Imagename = os.listdir(root_dir_zebra)

        self.length_dataset = max(len(self.horse_Imagename),len(self.zebra_Imagename))  #uneven length of dataset

        self.zebra_len = len(self.zebra_Imagename)
        self.horse_len = len(self.horse_Imagename)
    
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):

        zebra_img = self.zebra_Imagename[index % self.zebra_len]
        horse_img = self.horse_Imagename[index % self.horse_len]

        zebra_path = os.path.join(self.root_dir_zebra, zebra_img)
        horse_path = os.path.join(self.root_dir_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:

            augmentations = self.transform(image=zebra_img, image0=horse_img)

            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]


        return zebra_img, horse_img
    


