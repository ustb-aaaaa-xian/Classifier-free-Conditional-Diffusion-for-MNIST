import os
import string
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
class letters_dataloader(Dataset):
	def __init__(self,data_path,transform ,num_each_class:int = None):
		super(letters_dataloader).__init__()
		# 传过来一个datapath，
		self.transform = transform
		data_list = []
		self.num_each_class = num_each_class
		num_classes = [0 for _ in range(26)]
		# 可以先把大小写统一一下 A-Z:65-90,a-z:97-122
		for sub_dir in os.listdir(data_path):
			# img = cv2.imread(os.path.join(data_path,file_name))
			# img = img/255.
			asc_ = ord(sub_dir)
			if asc_ >=97:
				target = ord(sub_dir) - 97  #小写-97
			else:
				target = ord(sub_dir) - 65  #大写-65
			j = 0
			num_classes[target] += len(os.listdir(os.path.join(data_path, sub_dir)))
			for file_name in os.listdir(os.path.join(data_path, sub_dir)):
				j += 1
				if self.num_each_class != 0:
					if j > num_each_class:
						break
				img = Image.open(os.path.join(data_path, sub_dir, file_name))
				if self.transform is not None:
					img = self.transform(img)
				data_list.append((img, target))
				if num_each_class != 0:  # 少量样本情况下可以进行数据扩充，真实样本情况就不进行样本扩充了吧
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

class emnist_byclass_dataloader(Dataset):
	def __init__(self,data_path = "",transform = None,num_each_class  = 5000):
		super().__init__()
		self.transform = transform
		data_list = []
		self.num_each_class = num_each_class
		num_classes = [0 for _ in range(len(os.listdir(data_path)))]
		# 0-9,A-Z,a-z
		for sub_dir in os.listdir(data_path):
			# img = cv2.imread(os.path.join(data_path,file_name))
			# img = img/255.
			target = sub_dir.split('tensor(')[-1].split(')')[0] # str
			target = int(target)
			j = 0
			num_classes[target] += len(os.listdir(os.path.join(data_path, sub_dir)))
			for file_name in os.listdir(os.path.join(data_path, sub_dir)):
				j += 1
				if self.num_each_class != 0:
					if j > num_each_class:
						break
				img = Image.open(os.path.join(data_path, sub_dir, file_name))
				if self.transform is not None:
					img = self.transform(img)
				data_list.append((img, target))

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

class ocr_byclass_fientune(Dataset):
	def __init__(self, data_path, transform, num_each_class: int = None,expansion_ratio : int = None):
		super(ocr_byclass_fientune).__init__()
		# 传过来一个datapath，
		self.transform = transform
		data_list = []
		self.num_each_class = num_each_class
		num_classes = [0 for _ in range(62)]
		# ASCII: 0-9 :48-57 ;A-Z:65-90;a-z:97-122,将数据target进行对应
		for sub_dir in os.listdir(data_path):
			asc_ = ord(sub_dir)

			if asc_ >= 97:
				target = ord(sub_dir) - 97 + 36  # 小写 - 97 + 36 ; (36-61)
			elif asc_ >=65:
				target = ord(sub_dir) - 65 + 10  # 大写 - 65 + 10 ; (10-35)
			elif asc_>=48:
				target = ord(sub_dir) - 48       # 数字 0-9
			j = 0
			num_classes[target] += len(os.listdir(os.path.join(data_path, sub_dir)))
			for file_name in os.listdir(os.path.join(data_path, sub_dir)):
				j += 1
				if self.num_each_class != 0:
					if j > num_each_class:
						break
				img = Image.open(os.path.join(data_path, sub_dir, file_name))
				if self.transform is not None:
					img = self.transform(img)
				data_list.append((img, target))
				if expansion_ratio is not None:  # 少量样本情况下可以进行数据扩充，真实样本情况就不进行样本扩充了吧
					img1 = tf1(img)
					# img2 = tf2(img)
					img3 = tf3(img)
					img4 = tf4(img)
					img_1 = tf_1(img)
					img_2 = tf_2(img)
					img_3 = tf_3(img)
					current_data = [img1,img3,img4,img_1,img_2,img_3]
					expand_data = random.sample(current_data,expansion_ratio)
					for img_i in expand_data:
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
	def get_name_classes(self):
		name_classes = []
		c_exist = self.get_c_exist()
		for c in c_exist:
			if c in range(0,10):
				name_classes.append(str(c))
			elif c in range(10,35):
				name_classes.append(chr(c+65-10))
			elif c in range(35,62):
				name_classes.append(chr(c+97-35))
		print(name_classes)
		return name_classes


