import os
import random

import cv2
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms

tfs = []
# 传过来的tf有转单通道，Resize，ToTensor

# 先把旋转去掉，可能会漏边,randomcrop不行，因为数据本身就占了整个字符，不能再继续中心裁剪
# tf1 = transforms.RandomAffine(degrees =5,interpolation = transforms.InterpolationMode.BICUBIC)
tf1 = transforms.ColorJitter(brightness=0.1,contrast=0.1)
# tf2 = transforms.RandomRotation(degrees = 15)
tf3 = transforms.GaussianBlur(kernel_size = 5,sigma = (.1,.2))
tf4 = transforms.GaussianBlur(kernel_size = 3,sigma = (.1,.1))
# 加入以下几种，
tf_1 = transforms.RandomApply([tf1,tf3], p = 0.5)
tf_2 = transforms.RandomApply([tf1,tf4],p = 0.5)
tf_3 = transforms.RandomApply([tf3,tf4], p = 0.5)
class dataloader_v1(Dataset):
    def __init__(self,data_path,transform = None):
        super(dataloader_v1).__init__()
        self.data_list = []
        target , j = 0,0
        for file_name in os.listdir(data_path):
            # img = cv2.imread(os.path.join(data_path,file_name))
            # img = img/255.
            if j >=30:
                target += 1
                j = 0
            if target == 10 :
                break
            img = Image.open(os.path.join(data_path,file_name))
            self.data_list.append((img,target))
            self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data ,target = self.data_list[item]
        if self.transform != None:
            data = self.transform(data)
        return data,target

# 无数据扩增,适合用全部数据的情况
class dataloader_ocr(Dataset):
    def __init__(self, data_path, transform=None):
        super(dataloader_v1).__init__()
        # 传过来一个datapath，
        self.data_list = []
        for sub_dir in os.listdir(data_path):
            # img = cv2.imread(os.path.join(data_path,file_name))
            # img = img/255.
            target = int(sub_dir)
            j = 0
            for file_name in os.listdir(os.path.join(data_path,sub_dir)):
                j += 1
                img = Image.open(os.path.join(data_path,sub_dir, file_name))
                self.data_list.append((img, target))
                self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data, target = self.data_list[item]
        if self.transform != None:
            data = self.transform(data)
        return data, target

# 有数据扩增，手动设置原始数据集数量
class dataloader_ocr_color_transform(Dataset):
    def __init__(self,data_path,transform = None,num_each_class :int =  0):
        super(dataloader_v1).__init__()
        # 传过来一个datapath，
        self.transform = transform
        data_list = []
        self.num_each_class = num_each_class
        num_classes = [0 for _ in range(10)]
        for sub_dir in os.listdir(data_path):
            # img = cv2.imread(os.path.join(data_path,file_name))
            # img = img/255.
            target = int(sub_dir)
            j = 0
            num_classes[target] = len(os.listdir(os.path.join(data_path, sub_dir)))
            for file_name in os.listdir(os.path.join(data_path, sub_dir)):
                j += 1
                if self.num_each_class != 0:
                    if j>num_each_class:
                        break
                img = Image.open(os.path.join(data_path, sub_dir, file_name))
                if self.transform is not None:
                    img = self.transform(img)
                data_list.append((img, target))
                if num_each_class != 0: # 少量样本情况下可以进行数据扩充，真实样本情况就不进行样本扩充了吧
                    img1 = tf1(img)
                    # img2 = tf2(img)
                    img3 = tf3(img)
                    img4 = tf4(img)
                    img_1 = tf_1(img)
                    img_2 = tf_2(img)
                    img_3 = tf_3(img)
                    data_list.append((img1,target))
                    # data_list.append((img2,target))
                    data_list.append((img3,target))
                    data_list.append((img4,target))
                    data_list.append((img_1,target))
                    data_list.append((img_2,target))
                    data_list.append((img_3,target))

        self.data_list = data_list
        # 得到了self.data_list
        self.num_classes = num_classes
        print(f"number of classes :{self.num_classes}")
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data, target = self.data_list[item]
        return data, target

# 一个针对药盒数据的ocr
class dataloader_for_G716DVBX(Dataset):
    def __init__(self,data_path,transform = None,num_each_class:int = None):
        super().__init__()
        self.transform = transform
        data_list = []
        classes = ["0","1","2","5","6"] # 0_.1_
        for sub_dir in os.listdir(data_path):
            if sub_dir == "0_":
                target = 3
            elif sub_dir == "1_":
                target = 4
            else:
                target = int(sub_dir)
            j = 0
            for file_name in os.listdir(os.path.join(data_path, sub_dir)):
                j += 1
                if j > num_each_class:
                    break
                img = Image.open(os.path.join(data_path, sub_dir, file_name))
                if self.transform is not None:
                    img = self.transform(img)
                data_list.append((img, target))
                img1 = tf1(img)
                # img2 = tf2(img)
                img3 = tf3(img)
                img4 = tf4(img)
                img_1 = tf_1(img)
                img_2 = tf_2(img)
                img_3 = tf_3(img)
                data_list.append((img1, target))
                # data_list.append((img2,target))
                data_list.append((img3, target))
                data_list.append((img4, target))
                data_list.append((img_1, target))
                data_list.append((img_2, target))
                data_list.append((img_3, target))

        self.data_list = data_list
        # 得到了self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data, target = self.data_list[item]
        return data, target

class dataloader_for_masks(Dataset):
    def __init__(self,data_path,transform = None,num_each_class :int =  0,expansion_ratio : int = None):
        super(dataloader_v1).__init__()
        # 传过来一个datapath，
        self.transform = transform
        data_list = []
        self.num_each_class = num_each_class
        num_classes = [0 for _ in range(len(os.listdir(data_path)))]
        for sub_dir in os.listdir(data_path):
            target = int(sub_dir) - 1
            j = 0
            num_classes[target] = len(os.listdir(os.path.join(data_path, sub_dir)))
            for file_name in os.listdir(os.path.join(data_path, sub_dir)):
                j += 1
                if self.num_each_class != 0:
                    if j>num_each_class:
                        break
                img = Image.open(os.path.join(data_path, sub_dir, file_name))
                if self.transform is not None:
                    img = self.transform(img)
                data_list.append((img, target))
                if  expansion_ratio != 0: # 少量样本情况下可以进行数据扩充，真实样本情况就不进行样本扩充了吧
                    img1 = tf1(img)
                    # img2 = tf2(img)
                    img3 = tf3(img)
                    img4 = tf4(img)
                    img_1 = tf_1(img)
                    img_2 = tf_2(img)
                    img_3 = tf_3(img)
                    current_data = [img1,img3,img4,img_1,img_2,img_3]
                    if expansion_ratio >= (len(current_data)):
                        expansion_ratio = len(current_data)
                    expansion_data = random.sample(current_data,expansion_ratio)
                    for img_i in expansion_data:
                        data_list.append((img_i,target))

        self.data_list = data_list
        # 得到了self.data_list
        self.num_classes = num_classes
        print(f"number of classes :{self.num_classes}")
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data, target = self.data_list[item]
        return data, target


    def get_c_exist(self):
        c_exist = []
        n = len(self.num_classes)
        for i in range(n):
            if self.num_classes[i] != 0:
                c_exist.append(i)
        return c_exist



