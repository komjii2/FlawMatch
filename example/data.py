import torch
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from glob import glob
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCrackImageDataset(Dataset):
    def __init__(self, img_dir, train_test, label_num, channel, simple_transform=None, augmentation=False, wh = (64,64)):
        print("Data path load...")
        self.channel = channel
        self.label_num = label_num
        self.img_filenames = glob(os.path.join(img_dir, train_test,"*.png"))
        print("Data path sort...")
        self.img_filenames.sort()
        self.simple_transform = simple_transform
        self.wh = wh
        print("Data init end")
        
    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        min = -1
        max = 1
        ori_img = cv2.imread(self.img_filenames[idx],-1)
        ori_shape = ori_img.shape
        if self.channel == 1:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        ori_img = cv2.resize(ori_img,(self.wh[0],self.wh[1]),cv2.INTER_LANCZOS4)
        ori_img = (ori_img/255*(max-min))+min
        ori_img = ori_img.astype(np.float32)
        input_image = self.simple_transform(ori_img)
        
        base_name = os.path.basename(self.img_filenames[idx].replace(".png",""))
        base_name_split =   base_name.split("_")[-2:]
        wh_label_num = int(int(self.label_num)/100)
        lr_label_num = int(int(self.label_num)%100/10)
        tb_label_num = int(int(self.label_num)%10)
        if lr_label_num == 2:
            if base_name_split[0] == "right":
                x_flag = 1 #"right"
            else:
                x_flag = 0 #"left"
        else:
            if base_name_split[0] == "right":
                x_flag = 2 #"right"
            elif base_name_split[0] == "mid":
                x_flag = 1 #"mid"
            else:
                x_flag = 0 #"left"
        if tb_label_num == 2:
            if base_name_split[1] == "bottom":
                y_flag = int(1*lr_label_num) #"bottom"
            else:
                y_flag = int(0*lr_label_num) #"top"
        else:
            if base_name_split[1] == "bottom":
                y_flag = int(2*lr_label_num) #"bottom"
            elif base_name_split[1] == "mid":
                y_flag = int(1*lr_label_num) #"mid"
            else:
                y_flag = int(0*lr_label_num) #"top"
        if wh_label_num == 2:
            if ori_shape[1]/ori_shape[0] > 1: #horiszontal
                z_flag = int(1*lr_label_num*tb_label_num)
            else:
                z_flag = int(0*lr_label_num*tb_label_num)#vertical
        else:
            z_flag = 0
        #0-->Left Top           1-->Right Top           2-->Left Bottom     3-->Right Bottom
        #0-->Left Top           1-->Mid Top             2-->Right Top       3-->Left Mid        4-->Mid Mid        5-->Right Mide       ...
        label = torch.tensor(x_flag+y_flag+z_flag)
        return input_image, label