'''
This script does conditional image generation on MNIST, using a diffusion model
This code is modified from,
https://github.com/cloneofsimo/minDiffusion
Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487
'''
import sys
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST,EMNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from utils.letters_dataloader import emnist_byclass_dataloader, ocr_byclass_fientune
import sys
# sys.path.append("../utils")
from utils.dataloader import dataloader_ocr, dataloader_ocr_color_transform, dataloader_for_G716DVBX
import numpy as np
import os
import time
import argparse


class ResidualConvBlock(nn.Module):
	def __init__(
			self, in_channels: int, out_channels: int, is_res: bool = False
	) -> None:
		super().__init__()
		'''
		standard ResNet style convolutional block
		'''
		self.same_channels = in_channels == out_channels
		self.is_res = is_res
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, 1, 1),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, 3, 1, 1),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.is_res:
			x1 = self.conv1(x)
			x2 = self.conv2(x1)
			# this adds on correct residual in case channels have increased
			if self.same_channels:
				out = x + x2
			else:
				out = x1 + x2
			return out / 1.414  # 1.414是什么？
		else:
			x1 = self.conv1(x)
			x2 = self.conv2(x1)
			return x2


class UnetDown(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UnetDown, self).__init__()
		'''
		process and downscale the image feature maps
		'''
		layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


class UnetUp(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UnetUp, self).__init__()
		'''
		process and upscale the image feature maps
		'''
		layers = [
			nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
			ResidualConvBlock(out_channels, out_channels),
			ResidualConvBlock(out_channels, out_channels),
		]
		self.model = nn.Sequential(*layers)

	def forward(self, x, skip):
		x = torch.cat((x, skip), 1)
		x = self.model(x)
		return x


class EmbedFC(nn.Module):
	def __init__(self, input_dim, emb_dim):
		super(EmbedFC, self).__init__()
		'''
		generic one layer FC NN for embedding things  
		'''
		self.input_dim = input_dim
		layers = [
			nn.Linear(input_dim, emb_dim),
			nn.GELU(),
			nn.Linear(emb_dim, emb_dim),
		]
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(-1, self.input_dim)
		return self.model(x)


class ContextUnet(nn.Module):
	def __init__(self, in_channels, n_feat=256, n_classes=10):
		super(ContextUnet, self).__init__()

		self.in_channels = in_channels
		self.n_feat = n_feat
		self.n_classes = n_classes

		self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

		self.down1 = UnetDown(n_feat, n_feat)  # 4(batch) 128 14 14
		self.down2 = UnetDown(n_feat, 2 * n_feat)  # 4(batch) 256 7 7

		self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

		self.timeembed1 = EmbedFC(1, 2 * n_feat)
		self.timeembed2 = EmbedFC(1, 1 * n_feat)
		self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
		self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

		self.up0 = nn.Sequential(
			# nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
			nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
			nn.GroupNorm(8, 2 * n_feat),
			nn.ReLU(),
		)

		self.up1 = UnetUp(4 * n_feat, n_feat)
		self.up2 = UnetUp(2 * n_feat, n_feat)
		self.out = nn.Sequential(
			nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
			nn.GroupNorm(8, n_feat),
			nn.ReLU(),
			nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
		)

	def forward(self, x, c, t, context_mask):
		# x is (noisy) image, c is context label, t is timestep,
		# context_mask says which samples to block the context on

		x = self.init_conv(x)
		down1 = self.down1(x)
		down2 = self.down2(down1)
		hiddenvec = self.to_vec(down2)

		# convert context to one hot embedding
		c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

		# mask out context if context_mask == 1
		context_mask = context_mask[:, None]
		context_mask = context_mask.repeat(1, self.n_classes)
		context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
		c = c * context_mask

		# embed context, time step
		cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # 4 10
		temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)  # 4 256 1 1
		cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)  # 4 128 1 1
		temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

		# could concatenate the context embedding here instead of adaGN
		# hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

		up1 = self.up0(hiddenvec)
		# up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
		up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
		up3 = self.up2(cemb2 * up2 + temb2, down1)
		out = self.out(torch.cat((up3, x), 1))
		return out


def ddpm_schedules(beta1, beta2, T):
	"""
	Returns pre-computed schedules for DDPM sampling, training process.
	"""
	assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

	beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
	sqrt_beta_t = torch.sqrt(beta_t)
	alpha_t = 1 - beta_t
	log_alpha_t = torch.log(alpha_t)
	alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

	sqrtab = torch.sqrt(alphabar_t)
	oneover_sqrta = 1 / torch.sqrt(alpha_t)

	sqrtmab = torch.sqrt(1 - alphabar_t)
	mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

	return {
		"alpha_t": alpha_t,  # \alpha_t
		"oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
		"sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
		"alphabar_t": alphabar_t,  # \bar{\alpha_t}
		"sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
		"sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
		"mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
	}


class DDPM(nn.Module):
	def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
		super(DDPM, self).__init__()
		self.nn_model = nn_model.to(device)

		# register_buffer allows accessing dictionary produced by ddpm_schedules
		# e.g. can access self.sqrtab later
		for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
			self.register_buffer(k, v)

		self.n_T = n_T
		self.device = device
		self.drop_prob = drop_prob
		self.loss_mse = nn.MSELoss()

	def forward(self, x, c):
		"""
		this method is used in training, so samples t and noise randomly
		"""

		_ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
		noise = torch.randn_like(x)  # eps ~ N(0, 1)

		x_t = (
				self.sqrtab[_ts, None, None, None] * x
				+ self.sqrtmab[_ts, None, None, None] * noise
		)  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
		# We should predict the "error term" from this x_t. Loss is what we return.

		# dropout context with some probability
		context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

		# return MSE between added noise, and our predicted noise
		return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

	def sample(self, n_sample, size, device, guide_w=0.0):
		# we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
		# to make the fwd passes efficient, we concat two versions of the dataset,
		# one with context_mask=0 and the other context_mask=1
		# we then mix the outputs with the guidance scale, w
		# where w>0 means more guidance

		x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
		# c_i = torch.arange(0, 10).to(device)  # context for us just cycles throught the mnist labels
		c_i = torch.arange(0,62).to(device)
		c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

		# don't drop context at test time
		context_mask = torch.zeros_like(c_i).to(device)

		# double the batch
		c_i = c_i.repeat(2)
		context_mask = context_mask.repeat(2)
		context_mask[n_sample:] = 1.  # makes second half of batch context free

		x_i_store = []  # keep track of generated steps in case want to plot something
		print()
		for i in range(self.n_T, 0, -1):
			print(f'sampling timestep {i}', end='\r')
			t_is = torch.tensor([i / self.n_T]).to(device)
			t_is = t_is.repeat(n_sample, 1, 1, 1)

			# double batch
			x_i = x_i.repeat(2, 1, 1, 1)
			t_is = t_is.repeat(2, 1, 1, 1)

			z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

			# split predictions and compute weighting
			eps = self.nn_model(x_i, c_i, t_is, context_mask)
			eps1 = eps[:n_sample]
			eps2 = eps[n_sample:]
			eps = (1 + guide_w) * eps1 - guide_w * eps2  # eps2 是有条件模型的生成，eps1是无条件模型的生成
			x_i = x_i[:n_sample]
			x_i = (
					self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
					+ self.sqrt_beta_t[i] * z
			)
			if i % 20 == 0 or i == self.n_T or i < 8:
				x_i_store.append(x_i.detach().cpu().numpy())

		x_i_store = np.array(x_i_store)
		return x_i, x_i_store  # x_i_store就是所有x_I append之后变成ndarray,x_i就是最后一个
	def sample_for_exist(self, n_sample, size, device, guide_w=0.0,c_exist = []):

		# 加入一个新的参数c_exist
		x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
		c_i = torch.Tensor(c_exist) # context for us just cycles throught the exsit labels
		c_i = c_i.type(dtype = torch.int64).to(device)
		print(f"数据标签:c_i:{c_i},len(c_i):{len(c_i)}")
		c_i = c_i.repeat(int(n_sample / len(c_exist)))

		# don't drop context at test time
		context_mask = torch.zeros_like(c_i).to(device)

		# double the batch
		c_i = c_i.repeat(2)
		context_mask = context_mask.repeat(2)
		context_mask[n_sample:] = 1.  # makes second half of batch context free

		x_i_store = []  # keep track of generated steps in case want to plot something
		print()
		for i in range(self.n_T, 0, -1):
			print(f'sampling timestep {i:3d}', end='\r')
			t_is = torch.tensor([i / self.n_T]).to(device)
			t_is = t_is.repeat(n_sample, 1, 1, 1)

			# double batch
			x_i = x_i.repeat(2, 1, 1, 1)
			t_is = t_is.repeat(2, 1, 1, 1)

			z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

			# split predictions and compute weighting
			# print(f"x_i.shape:{x_i.shape},c_i.shape:{c_i.shape},t_is.shape:{t_is.shape},context_mask.shape:{context_mask.shape}")
			eps = self.nn_model(x_i, c_i, t_is, context_mask)
			eps1 = eps[:n_sample]
			eps2 = eps[n_sample:]
			eps = (1 + guide_w) * eps1 - guide_w * eps2  # eps2 是有条件模型的生成，eps1是无条件模型的生成
			x_i = x_i[:n_sample]
			x_i = (
					self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
					+ self.sqrt_beta_t[i] * z
			)
			if i % 20 == 0 or i == self.n_T or i < 8:
				x_i_store.append(x_i.detach().cpu().numpy())

		x_i_store = np.array(x_i_store)
		return x_i, x_i_store  # x_i_store就是所有x_I append之后变成ndarray,x_i就是最后一个
	def sample_for_paticular_c(self,n_sample, size, device, guide_w=0.0,c = []):


		# 加入一个新的参数c_exist
		x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
		c_i = torch.ones(n_sample) * c # context for us just cycles throught the exsit labels
		c_i = c_i.type(dtype = torch.int64).to(device)
		print(f"数据标签:c_i:{c_i},len(c_i):{len(c_i)}")

		# don't drop context at test time
		context_mask = torch.zeros_like(c_i).to(device)

		# double the batch
		c_i = c_i.repeat(2)
		context_mask = context_mask.repeat(2)
		context_mask[n_sample:] = 1.  # makes second half of batch context free

		x_i_store = []  # keep track of generated steps in case want to plot something
		print()
		for i in range(self.n_T, 0, -1):
			print(f'sampling timestep {i:3d}', end='\r')
			t_is = torch.tensor([i / self.n_T]).to(device)
			t_is = t_is.repeat(n_sample, 1, 1, 1)

			# double batch
			x_i = x_i.repeat(2, 1, 1, 1)
			t_is = t_is.repeat(2, 1, 1, 1)

			z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

			# split predictions and compute weighting
			# print(f"x_i.shape:{x_i.shape},c_i.shape:{c_i.shape},t_is.shape:{t_is.shape},context_mask.shape:{context_mask.shape}")
			eps = self.nn_model(x_i, c_i, t_is, context_mask)
			eps1 = eps[:n_sample]
			eps2 = eps[n_sample:]
			eps = (1 + guide_w) * eps1 - guide_w * eps2  # eps2 是有条件模型的生成，eps1是无条件模型的生成
			x_i = x_i[:n_sample]
			x_i = (
					self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
					+ self.sqrt_beta_t[i] * z
			)
			if i % 20 == 0 or i == self.n_T or i < 8:
				x_i_store.append(x_i.detach().cpu().numpy())

		x_i_store = np.array(x_i_store)
		return x_i, x_i_store  # x_i_store就是所有x_I append之后变成ndarray,x_i就是最后一个


def emnist_byclass_pretrain():
	n_epoch = create_parser().n_epoch
	batch_size = create_parser().batch_size
	n_T = create_parser().n_T
	device = create_parser().device
	n_classes = create_parser().n_classes
	n_feat = create_parser().n_feat
	lrate = create_parser().lrate


	ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

	ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
				device=device, drop_prob=0.1)  # 注意通道数的改变
	# 预训练无加载模型，随机参数初始化
	ddpm.to(device)
	ddpm.load_state_dict(torch.load("./data/pretrain_byclass_letters/model_40.pth"))
	print(f"load pth_model done !")
	tf = transforms.Compose([
		transforms.ToTensor(),
		transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 图片三通道不能reoeat
		transforms.Resize((28, 28)),
		transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
	])
	save_dir = f'./data/pretrain_byclass_letters_/'  # emnist_letters预训练保存
	if not os.path.exists(save_dir):
		print("save_path not exists,try to mkdirs")
		os.makedirs(save_dir)
	# data
	# dataset = EMNIST(root = "./emnist",split = "byclass",transform = tf,train = True,download = True)
	dataset = emnist_byclass_dataloader(data_path = "./data/emnist_byclass_balanced1000",transform = tf,num_each_class = 1000)
	print(f"dataset[0][0].shape:{dataset[0][0].shape}")
	c_exist = dataset.get_c_exist()
	images_real_show = {label:[] for label in c_exist} #记录的是Tensor以及标签 可以将字典里面的键设置成label【int】，值设置成列表，是Tensor的列表
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
	print(len(dataset),type(dataset),len(dataloader))
	len_data = len(dataset)
	for i in range(len_data):
		if type(dataset[i]) != tuple or len(dataset[i] )!= 2:
			print(f"dataset[{i}] is not tuple or  its len is {len(dataset[i])}.")
	for ep in range(n_epoch):

		print(f'epoch {ep}')
		ddpm.train()

		# linear lrate decay
		optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

		pbar = tqdm(dataloader)
		loss_ema = None
		for (x,c) in pbar:

			optim.zero_grad()
			# 写一个找到全部类别真实数据的函数并赋给x_real
			# 判断数据是否够用
			flag = 1  # flag 是判断数据是否够用的标志，1代表够用，判断是否够用，不够用就置0
			for label,datas in images_real_show.items():
				if len(datas) < create_parser().n_sample_mul:
					flag = 0

			if flag ==0 :
				batch_ = x.shape[0]
				for i in range(batch_):
					# 都是Tensor
					data =  x[i]
					label = c[i]
					if len(images_real_show[int(label)])< create_parser().n_sample_mul:
						images_real_show[int(label)].append(data)
		# 	# 这句其实不需要了，因为数据集是自己手动调整好的，不需要再transpose
			x = x.to(device)
			c = c.to(device)
			loss = ddpm(x, c)
			loss.backward()
			if loss_ema is None:
				loss_ema = loss.item()
			else:
				loss_ema = (n_epoch-1) / (n_epoch) * loss_ema + 1 / n_epoch  * loss.item()
			pbar.set_description(f"loss: {loss_ema:.4f}")
			optim.step()

		# for eval, save an image of currently generated samples (top rows)
		# followed by real images (bottom rows)
		end_time1 = time.time()

		ddpm.eval()
		if not (ep % 5 ==0  or ep >=(n_epoch -1)) :
			continue
		with torch.no_grad():
			n_sample = create_parser().n_sample_mul * n_classes
			if True:
				for w_i, w in enumerate(ws_test):
					if w != 2.0:
						continue
					x_gen, x_gen_store = ddpm.sample_for_exist(n_sample, (3, 28, 28), device, guide_w=w,
															   c_exist=c_exist)

					# append some real images at bottom, order by class also
					x_real = torch.Tensor(x_gen.shape).to(device)
					# n_, c_, h_, w_ = x_gen.shape
					# x_real = torch.Tensor(size=(2 * n_classes, c_, h_, w_)).to(device)
					print(f"x_real.shape:{x_real.shape}")

					# 通过在上面得到的image_real_show来展示对应标签的真实数据
					for k in range(len(c_exist)):
						for j in range(int(x_real.shape[0] // len(c_exist))):
							x_real[k + j * len(c_exist)] = images_real_show[c_exist[k]][j]
					x_gen = (x_gen - x_gen.min())/(x_gen.max() - x_gen.min())
					x_all = torch.cat([x_gen, x_real])

					# 如果前面将正的图像进行transpose之后才放入网络的，所以生成的都是不正的，其实可以将生成的图像进行transpose一下，应该是可以展示成正的图像的。

					grid = make_grid((x_all - x_all.min()) / (x_all.max() - x_all.min()),
									 nrow=len(c_exist))  # n_row改成n_classes
					save_image(grid, save_dir + f"emnist_byclass40_image_ep{ep}_w{w}.png")
					print('saved image at ' + save_dir + f"emnist_byclass40_image_ep{ep}_w{w}.png")

					# 动图的保存又耗时间，又耗内存，只在最后一个epoch保存就好
					if ep > int(n_epoch - 1):
						# create gif of images evolving over time, based on x_gen_store
						fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, sharex=True,
												sharey=True,
												figsize=(8, 3))

						def animate_diff(i, x_gen_store):
							print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
							plots = []
							for row in range(int(n_sample / n_classes)):
								for col in range(n_classes):
									axs[row, col].clear()
									axs[row, col].set_xticks([])
									axs[row, col].set_yticks([])
									# plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
									plots.append(
										axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], cmap='gray',
															 vmin=(-x_gen_store[i]).min(),
															 vmax=(-x_gen_store[i]).max()))
							return plots

						ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False,
											repeat=True,
											frames=x_gen_store.shape[0])
						ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
						print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
		if ep % 5 == 0 or ep == (n_epoch - 1):  # optionally save model
			torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
			print('saved model at ' + save_dir + f"model_{ep}.pth")

def ocrall_byclass_finetune():

	arg = create_parser()
	n_epoch = arg.n_epoch
	batch_size = arg.batch_size
	n_T = arg.n_T
	device = arg.device
	n_classes = arg.n_classes
	n_feat = arg.n_feat
	lrate = arg.lrate
	ocr = arg.ocr
	dataset_name = arg.dataset_name
	num_each_class = arg.num_each_class
	n_sample_mul = arg.n_sample_mul
	n_real_mul = arg.n_real_mul
	height = arg.h
	width = arg.w
	expansion_ratio = arg.expansion_ratio

	ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

	ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
				device=device, drop_prob=0.1)  # 注意通道数的改变
	# 预训练无加载模型，随机参数初始化
	ddpm.to(device)
	ddpm.load_state_dict(torch.load("./data/pretrain_byclass_letters_/model_15.pth"))
	tf = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((28, 28)),
		# transforms.Normalize([.5,.5,.5],[.5,.5,.5]), # 不是识别任务没必要吧，又不是为了更好的分类

	])
	save_dir = f'./Generate_save/ocr_byclass_finetune/ocr{ocr}_/{dataset_name}_{num_each_class}/'  # emnist_letters预训练保存
	if not os.path.exists(save_dir):
		print("save_path not exists,try to mkdirs")
		os.makedirs(save_dir)
	# data
	# dataset = EMNIST(root = "./emnist",split = "byclass",transform = tf,train = True,download = True)
	dataset = ocr_byclass_fientune(data_path = f"./data/ocr{ocr}/{dataset_name}_ocr_all",transform = tf,num_each_class = num_each_class,expansion_ratio = expansion_ratio)
	c_exist = dataset.get_c_exist()
	print(f"c_exist:{c_exist}")
	images_real_show = {label:[] for label in c_exist} #记录的是Tensor以及标签 可以将字典里面的键设置成label【int】，值设置成列表，是Tensor的列表
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
	print(len(dataset),type(dataset),len(dataloader))
	len_data = len(dataset)
	for i in range(len_data):
		if type(dataset[i]) != tuple or len(dataset[i] )!= 2:
			print(f"dataset[{i}] is not tuple or  its len is {len(dataset[i])}.")
	for ep in range(n_epoch):

		print(f'epoch {ep}')
		ddpm.train()

		# linear lrate decay
		optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

		pbar = tqdm(dataloader)
		loss_ema = None
		for (x,c) in pbar:

			optim.zero_grad()
			# 写一个找到全部类别真实数据的函数并赋给x_real
			# 判断数据是否够用
			flag = 1  # flag 是判断数据是否够用的标志，1代表够用，判断是否够用，不够用就置0
			for label,datas in images_real_show.items():
				# 某个类别展示的不够，flag置0
				if len(datas) < n_real_mul:
					flag = 0

			if flag == 0 :
				batch_ = x.shape[0]
				for i in range(batch_):
					# 都是Tensor
					data = x[i]
					label = c[i]
					# 不够的数据才继续加入
					if len(images_real_show[int(label)])< n_real_mul:
						images_real_show[int(label)].append(data)
			x = x.to(device)
			c = c.to(device)
			loss = ddpm(x, c)
			loss.backward()
			if loss_ema is None:
				loss_ema = loss.item()
			else:
				loss_ema = (n_epoch-1) / (n_epoch) * loss_ema + 1 / n_epoch  * loss.item()
			pbar.set_description(f"loss: {loss_ema:.4f}")
			optim.step()

		# for eval, save an image of currently generated samples (top rows)
		# followed by real images (bottom rows)

		ddpm.eval()
		if not (ep % 50 ==0  or ep >=n_epoch -1 )  : # 手动调整进行保存的epoch
			continue
		with torch.no_grad():
			n_sample = n_sample_mul * len(c_exist)
			# 展示拼接图像
			if True:
				for w_i, w in enumerate(ws_test):
					if w != 2.0:
						continue
					generate_s_time = time.time()
					x_gen, x_gen_store = ddpm.sample_for_exist(n_sample, (3, 28, 28), device, guide_w=w,
															   c_exist=c_exist)
					generate_e_time = time.time()
					print(f"{generate_e_time - generate_s_time} has passed when generating {n_sample} images.")
					# append some real images at bottom, order by class also
					# x_real = torch.Tensor(x_gen.shape).to(device)
					n_, c_, h_, w_ = x_gen.shape
					x_real = torch.Tensor(size=( n_real_mul * len(c_exist), c_, h_, w_)).to(device)
					print(f"x_real.shape:{x_real.shape}")

					# 通过在上面得到的image_real_show来展示对应标签的真实数据
					for k in range(len(c_exist)):
						for j in range(int(x_real.shape[0] // len(c_exist))):
							x_real[k + j * len(c_exist)] = images_real_show[c_exist[k]][j]
					x_gen = (x_gen - x_gen.min()) / (x_gen.max() - x_gen.min())
					x_real = (x_real - x_real.min()) / (x_real.max() - x_real.min())
					x_all = torch.cat([x_gen, x_real])

					# 如果前面将正的图像进行transpose之后才放入网络的，所以生成的都是不正的，其实可以将生成的图像进行transpose一下，应该是可以展示成正的图像的。
					grid = make_grid((x_all - x_all.min()) / (x_all.max() - x_all.min()),
									 nrow=len(c_exist))
					save_image(grid, save_dir + f"{dataset_name}_image_ep{ep}_w{w:.1f}.png")
					print('saved image at ' + save_dir + f"{dataset_name}_image_ep{ep}_w{w:.1f}.png")
					generate_e_time = time.time()
					print(f"{generate_e_time - generate_s_time} has passed when generating and saving {n_sample} images.")
					# 动图的保存又耗时间，又耗内存，只在最后一个epoch保存就好(或者不保存)
					if ep > int(n_epoch - 1):
						# create gif of images evolving over time, based on x_gen_store
						fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, sharex=True,
												sharey=True,
												figsize=(8, 3))

						def animate_diff(i, x_gen_store):
							print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
							plots = []
							for row in range(int(n_sample / n_classes)):
								for col in range(n_classes):
									axs[row, col].clear()
									axs[row, col].set_xticks([])
									axs[row, col].set_yticks([])
									# plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
									plots.append(
										axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], cmap='gray',
															 vmin=(-x_gen_store[i]).min(),
															 vmax=(-x_gen_store[i]).max()))
							return plots

						ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False,
											repeat=True,
											frames=x_gen_store.shape[0])
						ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
						print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
			# 对生成的单字符进行保存
			if ep == (n_epoch - 1):
				generate_save_dir = save_dir
				height, width = arg.h, arg.w  # 直接写在parser里
				for w_i, w in enumerate(ws_test):  # 如果只想生成 w = 2.0 的话就可以不用for循环
					if w != 2.0:
						continue
					generate_s_time = time.time()
					x_gen, x_gen_store = ddpm.sample_for_exist(n_sample, (3, 28, 28), device,
													 guide_w=w,c_exist=c_exist)  # 这个sample函数是不是也可以重写，因为根本不需要生成过程的保存，只需要保存生成效果(没必要，生成中间结果的保存并不占用时间)

					x_ = []
					# 得到类别名称
					x_gen = (x_gen - x_gen.min()) /(x_gen.max() - x_gen.min())
					name_classes = dataset.get_name_classes()
					print(f"name_classes:{name_classes}")
					for i in range(x_gen.shape[0]):
						img = x_gen[i]
						img = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC)(
							img)  # 还原形状保存
						j = 0
						save_path = generate_save_dir + f"{dataset_name}_image_w{w:.1f}_{name_classes[i % len(c_exist)]}_{j}.png"
						while os.path.exists(save_path):
							j += 1
							save_path = f"{generate_save_dir}{dataset_name}_image_w{w:.1f}_{name_classes[i % len(c_exist)]}_{j}.png"
						save_image(img, save_path)
						print('saved image at ' + save_path, end="\r")
					generate_e_time = time.time()
					print(
						f"{generate_e_time - generate_s_time}s has passed when generating {n_sample_mul} batches of {n_sample} characters of ocr{ocr} model_id {dataset_name},w={w:.1f}.")
			# 生成特定类别的单字符
			if ep == (n_epoch - 1):
				x_gen, x_gen_store = ddpm.sample_for_paticular_c(n_sample=n_sample, size=(3, 28, 28), device=device,
															   guide_w=w,
															   c =c_exist[-2])

				x_ = []
				# j = 0
				for i in range(x_gen.shape[0]):
					img = x_gen[i]
					if height < 28 or width < 28:
						img = transforms.Resize((height, width),
												interpolation=transforms.InterpolationMode.NEAREST)(
							img)
					else:
						img = transforms.Resize((height, width),
												interpolation=transforms.InterpolationMode.BICUBIC)(
							img)  # 变成原始长条状
					j = 0
					save_path = generate_save_dir + f"image_w{w:.1f}_{name_classes[-2]}_{j}.png"
					while os.path.exists(save_path):
						j += 1
						save_path = f"{generate_save_dir}image_w{w:.1f}_{name_classes[-2]}_{j}.png"
					img = (img - img.min()) / (img.max() - img.min())
					save_image(img, save_path)
					print('saved image at ' + save_path, end="\r")
				generate_e_time = time.time()
				print(
					f"{generate_e_time - generate_s_time}s has passed when generating and saving {n_sample} characters.")
		if ep == (n_epoch - 1) :  # optionally save model
			torch.save(ddpm.state_dict(), save_dir + f"{dataset_name}_model_{ep}.pth")
			print('saved model at ' + save_dir + f"{dataset_name}_model_{ep}.pth")
def create_parser():  # 这样写是可行的
	parser = argparse.ArgumentParser()

	parser.add_argument("--n_epoch", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--n_T", type=int, default=400)
	parser.add_argument("--n_classes", type=int, default=62)
	parser.add_argument("--dataset_name",type = str,default = "S1ONVPHW")
	parser.add_argument("--n_feat", type=int, default=128)
	parser.add_argument("--lrate", type=float, default=1e-4)
	# parser.add_argument("--save_dir", type=str, default="./output/")
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--ocr", type=str, default='4', help="the number of data batch")
	parser.add_argument("--n_sample_mul",type = int ,default = 4) #生成图像的个数
	parser.add_argument("--n_real_mul",type = int ,default = 2) # 展示真实图像的个数
	parser.add_argument("--num_each_class",type = int ,default = 5)
	parser.add_argument("--h",type = int ,default = 45)
	parser.add_argument("--w",type = int ,default = 25)
	parser.add_argument("--expansion_ratio",type = int ,default = 6)

	arg = parser.parse_args()
	return arg
def get_parameter_number(model):
	total_num = sum(p.numel() for p in model.parameters())
	trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
	s_time = time.time()
	ocrall_byclass_finetune()
	end_time = time.time()
	print(f"{end_time - s_time}s has passed after finetune emnist_letters for {create_parser().n_epoch} epochs and all the generating and svaing eperations.")
	# n_feat = 128
	# n_classes = 62
	# device = "cuda:0"
	# n_T = 400
	# model = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
	# 			device=device, drop_prob=0.1)  # 注意通道数的改变
	# a = get_parameter_number(model) #{'Total': 6610179, 'Trainable': 6610179}
	# print(a)
