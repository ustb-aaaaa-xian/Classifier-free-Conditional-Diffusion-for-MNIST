from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from utils.dataloader import dataloader_ocr,dataloader_ocr_color_transform,dataloader_for_G716DVBX,dataloader_for_masks
import numpy as np
import os
import time
import argparse
class Conv3(nn.Module):
	def __init__(self,in_channels,out_channels,is_res : bool = False):
		super().__init__()
		self.same_channels = (in_channels == out_channels)
		self.main = nn.Sequential(
			nn.Conv2d(in_channels,out_channels,3,1,1),
			nn.GroupNorm(8,out_channels),
			nn.ReLU(inplace= True) # 激活函数用ReLU还是GeLU
		)
		self.conv = nn.Sequential(
			nn.Conv2d(out_channels,out_channels,3,1,1),
			nn.GroupNorm(8,out_channels),
			nn.ReLU(inplace = True),
		)
		self.is_res = is_res
	def forward(self,x):
		x1 = self.main(x)
		x2 = self.conv(x1)
		if self.is_res :
			if self.same_channels:
				return (x + x2)/1.414
			else:
				return (x1 + x2)/1.414
		else:
			return x2
class UNetDown(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layers = [Conv3(in_channels,out_channels),nn.MaxPool2d(2)]
		self.model = nn.Sequential(*self.layers)
	def forward(self,x):
		return self.model(x)

class UNetUp(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.layers = [
			nn.ConvTranspose2d(in_channels,out_channels,2,2),
			Conv3(out_channels,out_channels),
			Conv3(out_channels,out_channels)
		]

		self.model = nn.Sequential(*self.layers)
	def forward(self,x : torch.Tensor , skip : torch.Tensor) -> torch.Tensor :
		x = torch.cat([x,skip],dim=1)
		x = self.model(x)
		return x

class EmbedFC(nn.Module):
	def __init__(self,input_dim,out_dim):
		super().__init__()
		self.input_dim = input_dim
		self.layers =[
			nn.Linear(input_dim,out_dim),
			nn.GELU(),
			nn.Linear(out_dim,out_dim)
		]
		self.model = nn.Sequential(*self.layers)
	def forward(self,x):
		x = x.view(-1,self.input_dim)
		return self.model(x)

class ContextUnet(nn.Module):
	def __init__(self,in_channels,n_feat : int  = 256,n_classes = 10):
		super().__init__()

		self.n_feat = n_feat
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.init_conv = Conv3(in_channels,n_feat,False)  # -1 * emb_dim * 128 * 128
		self.down1 = UNetDown(n_feat,n_feat)  # -1 * emb_dim * 64 * 64
		self.down2 = UNetDown(n_feat,2*n_feat)  # -1 * emb_dim * 2 * 32 * 32
		self.down3 = UNetDown(2*n_feat,2*n_feat) # -1 * emb_dim * 2 * 16 * 16
		self.to_vec = nn.AvgPool2d(kernel_size = 4 ) # -1 * emb_dim * 2 * 4 * 4

		self.up0 = nn.Sequential(
			nn.ConvTranspose2d(n_feat *2 ,n_feat * 2,4,4),
			nn.GroupNorm(8,n_feat * 2),
			nn.ReLU(inplace = True)
		)
		self.up1 = UNetUp(4 * n_feat,2 * n_feat)
		self.up2 = UNetUp(4 * n_feat, n_feat)
		self.up3 = UNetUp(2 * n_feat,1 * n_feat)
		self.out = nn.Sequential(
			nn.Conv2d(2 * n_feat,n_feat, 3, 1, 1),
			nn.GroupNorm(8,n_feat),
			nn.ReLU(),
			nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
		)

		self.timeembed1 = EmbedFC(1,  2 * self.n_feat)
		self.timeembed2 = EmbedFC(1,  2 * self.n_feat)

		self.contextembed1 = EmbedFC(n_classes, 2 * self.n_feat)
		self.contextembed2 = EmbedFC(n_classes, 2 * self.n_feat)

	def forward(self,x:torch.Tensor,c,t:torch.Tensor,context_mask) -> torch.Tensor:
		x = self.init_conv(x) # n_feat

		# downsample
		down1 = self.down1(x) # n_feat * 2
		down2 = self.down2(down1) #n_feat *4
		down3 = self.down3(down2) #n_feat *8
		hiddenvec = self.to_vec(down3) # n_feat*8

		c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float) #有类别引导，要不先把这个类别引导去掉？？
		context_mask = context_mask[:, None]
		context_mask = context_mask.repeat(1, self.n_classes)
		context_mask = (-1 * (1 - context_mask))  # need to flip 0 <-> 1
		c = c * context_mask
		# embed context
		cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
		cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)

		temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)  #如果要用linear函数的话
		temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)

		# upsample
		up0 = self.up0(hiddenvec) # 16 * 16
		up1 = self.up1(cemb1 * up0 + temb1, down3) # 32 * 32
		up2 = self.up2(cemb2 * up1 + temb2, down2) # 64 * 64
		up3 = self.up3(up2 ,down1) # 128 * 128 * n_feat
		out = self.out(torch.cat((up3,x),dim =1))
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
		c_i = torch.arange(0, 10).to(device)  # context for us just cycles throught the mnist labels
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
			print(f'sampling timestep {i:3d}', end='\r')
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
			eps = (1 + guide_w) * eps1 - guide_w * eps2 # eps2 是有条件模型的生成，eps1是无条件模型的生成
			x_i = x_i[:n_sample]
			x_i = (
					self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
					+ self.sqrt_beta_t[i] * z
			)
			if i % 20 == 0 or i == self.n_T or i < 8:
				x_i_store.append(x_i.detach().cpu().numpy())

		x_i_store = np.array(x_i_store)
		return x_i, x_i_store  # x_i_store就是所有x_I append之后变成ndarray,x_i就是最后一个


	def sample_for_exist(self, n_sample, size, device, guide_w=0.0, c_exist=[]):
		# 加入一个新的参数c_exist
		x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
		c_i = torch.Tensor(c_exist)  # context for us just cycles throught the exsit labels
		c_i = c_i.type(dtype=torch.int64).to(device)
		print(f"数据标签:c_i{c_i}")
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

def masks_finetune64():
	arg = create_parser()
	start_finetune_time = time.time()
	n_epoch = arg.n_epoch
	batch_size = arg.batch_size
	n_T = arg.n_T
	device = arg.device
	n_classes = arg.n_classes
	n_feat = arg.n_feat
	lrate = arg.lrate
	dataset_name = arg.dataset_name
	num_each_class = arg.num_each_class
	n_sample_mul = arg.n_sample_mul
	n_real_mul = arg.n_real_mul

	save_dir = f'./output/mask/{dataset_name}/{dataset_name}_cifar64/'
	ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

	ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
				device=device, drop_prob=0.1)  # 注意通道数的改变
	# ddpm.load_state_dict(torch.load('./output/diffusion_outputs_ocr_finetune_v1_extreme_data_augment_YQCSSZ18/model_45.pth'))
	# 如果有分辨率的变化应该注意网络结构是否变化
	# 尝试无预训练模型
	ddpm.load_state_dict(torch.load("./data/diffusion_output_Unet64_cifar10_conditional_255/model_39.pth",map_location = "cuda:0"))
	# ddpm.load_state_dict(torch.load(save_dir + 'model_130.pth'))
	ddpm.to(device)
	# print(ddpm)
	tf = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Lambda(lambda x:x.repeat(3,1,1)), # 灰度图model_id :S1ONVPHW原始图象是bmp图像的单通道格式
		transforms.Resize((64,64))

	])  # mnist is already normalised 0 to 1

	if not os.path.exists(save_dir):
		print("save_path not exists,try to mkdirs")
		os.makedirs(save_dir)
	# 尝试多种数据，改写每次的dataset
	dataset = dataloader_for_masks(data_path=f"./data/anomaly/{dataset_name}/{dataset_name}_masks_padding", transform=tf,num_each_class=num_each_class, expansion_ratio=0)
	# 得到新c_exist
	c_exist = dataset.get_c_exist()  # 返回的是列表
	len_real = len(c_exist)
	print(f"len(real):{len_real}")
	images_real_show = {label: [] for label in c_exist}  # 记录的是Tensor以及标签 可以将字典里面的键设置成label【int】，值设置成列表，是Tensor的列表
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

	for ep in range(n_epoch):
		print(f'epoch {ep}')
		ddpm.train()

		# linear lrate decay
		optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

		pbar = tqdm(dataloader)
		loss_ema = None
		for x, c in pbar:  # x是data,c是label;训练/微调
			optim.zero_grad()
			# 写一个找到全部类别真实数据的函数并赋给x_real
			# 判断数据是否够用
			flag = 1  # flag 是判断数据是否够用的标志，1代表够用，判断是否够用，不够用就置0
			for label, datas in images_real_show.items():
				if len(datas) < n_real_mul: # 此处如果是小于create_parser().n_sample_mull就是real和gen展示数量一样，但是其实最好别
					flag = 0

			if flag == 0:
				batch_ = x.shape[0]
				for i in range(batch_):
					# 都是Tensor
					data = x[i]
					label = c[i]
					if len(images_real_show[int(label)]) < n_real_mul : #
						images_real_show[int(label)].append(data)

			# 经过以上步骤得到了image_real_show的一个字典，里面记录了真实标签下的真实数据

			x = x.to(device)
			c = c.to(device)
			loss = ddpm(x, c)
			loss.backward()
			if loss_ema is None:
				loss_ema = loss.item()
			else:
				loss_ema = (n_epoch - 1) / n_epoch * loss_ema + 1 / n_epoch * loss.item()  # 这个是根据20个epoch来的,所以要更改epoch比例
			pbar.set_description(f"loss: {loss_ema:.4f}")
			optim.step()

		# for eval, save an image of currently generated samples (top rows)
		# followed by real images (bottom rows)

		ddpm.eval()
		if (not (ep % 50 == 0 or ep == (n_epoch - 1 ))) or  ep==0:
			continue
		if ep == n_epoch - 1:
			end_finetune_time = time.time()
			print(f"after {n_epoch} epochs finetune ,{end_finetune_time - start_finetune_time} has passed.")
		with torch.no_grad():
			n_sample = n_sample_mul * len(c_exist)
			# 这个for循环是生成以及真实都保存了，再写一个直接生成的函数
			if True:
				for w_i, w in enumerate(ws_test):
					if w !=2.0:
						continue
					generate_s_time = time.time()
					x_gen, x_gen_store = ddpm.sample_for_exist(n_sample, (3,64,64), device, guide_w=w,
															   c_exist=c_exist)

					# append some real images at bottom, order by class also
					# x_real = torch.Tensor(x_gen.shape).to(device)
					n_, c_, h_, w_ = x_gen.shape
					x_real = torch.Tensor(size=(n_real_mul * len(c_exist), c_, h_, w_)).to(device)
					print(f"the labels of the last batch:{c}")  # 打印最后的标签

					# 通过在上面得到的image_real_show来展示对应标签的真实数据
					for k in range(len(c_exist)):
						for j in range(n_real_mul):
							x_real[k + j * len(c_exist)] = images_real_show[c_exist[k]][j]
					# 生成图像统一到0-1，展示的时候real不会随着
					x_gen = (x_gen - x_gen.min()) / (x_gen.max() - x_gen.min())
					x_real = (x_real - x_real.min()) / (x_real.max() - x_real.min())
					x_all = torch.cat([x_gen, x_real])
					print(f"x_all.min():{x_all.min()},x_all.max():{x_all.max()},x_real.min():{x_real.min()},x_real.max():{x_real.max()}")

					grid = make_grid((x_all - x_all.min()) / (x_all.max() - x_all.min()),
									 nrow=len(c_exist))
					save_image(grid, save_dir + f"{dataset_name}_image_ep{ep}_w{w}.png")
					print('saved image at ' + save_dir + f"{dataset_name}_image_ep{ep}_w{w}.png")
					generate_e_time = time.time()
					print(f"{generate_e_time - generate_s_time} has passed when generating and saving "
						  f"{n_sample} images.")

					if ep > int(n_epoch - 1):  # 不进行动图保存
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
			# 不记录动图，以及真实图片，只进行生成图片的保存，以及时间的记录.【也可以进行生成图片与真实图片的拼接，查看生成多样性的对比】
			if ep == (n_epoch - 1 ):
				generate_save_dir = save_dir
				height, width = create_parser().h, create_parser().w
				for w_i, w in enumerate(ws_test):  # 如果只想生成 w = 2.0 的话就可以不用for循环
					if w != 2.0:
						continue
					generate_s_time = time.time()
					x_gen, x_gen_store = ddpm.sample_for_exist(n_sample, (3,64,64), device,
															   guide_w=w,
															   c_exist=c_exist)  # 这个sample函数是不是也可以重写，因为根本不需要生成过程的保存，只需要保存生成效果(没必要，生成中间结果的保存并不占用时间)


					x_gen = (x_gen - x_gen.min()) / (x_gen.max() - x_gen.min())
					x_ = []
					# j = 0
					for i in range(x_gen.shape[0]):
						img = x_gen[i]
						if height < 64 or width < 64:
							img = transforms.Resize((height, width),
													interpolation=transforms.InterpolationMode.NEAREST)(
								img)
						else:
							img = transforms.Resize((height, width),
													interpolation=transforms.InterpolationMode.BICUBIC)(
								img)  # 变成原始长条状
						# 新加tf_Normalize还原原始均值以及亮度

						j = 0
						save_path = generate_save_dir + f"{dataset_name}_image_w{w:.1f}_{i%len(c_exist)}_{j}.png"
						while os.path.exists(save_path):
							j += 1
							save_path = generate_save_dir + f"{dataset_name}_image_w{w:.1f}_{i%len(c_exist)}_{j}.png"
						img = (img - img.min())/(img.max() - img.min())
						save_image(img, save_path)
						print('saved image at ' + save_path, end="\r")
					generate_e_time = time.time()
					print(
						f"{generate_e_time - generate_s_time}s has passed when generating {n_sample // len(c_exist)} batches of {n_sample} masks of  model_id {dataset_name}.")

		if (ep == (n_epoch - 1) or ep % 50 == 0) and ep != 0 :  # optionally save model
			torch.save(ddpm.state_dict(), save_dir + f"{dataset_name}_model_{ep}.pth")
			print('saved model at ' + save_dir + f"{dataset_name}_model_{ep}.pth")


# 再写一个train_data_time 主要记录时间【100个epoch微调以及每个batch的生成时间】
def create_parser():  # 这样写是可行的
	parser = argparse.ArgumentParser()

	parser.add_argument("--n_epoch", type=int, default=200)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--n_T", type=int, default=400)
	parser.add_argument("--n_classes", type=int, default=10)
	parser.add_argument("--n_feat", type=int, default=128)
	parser.add_argument("--lrate", type=float, default=3e-4)
	parser.add_argument("--dataset_name", default="T6J9BF9K", type=str)  # 数据集名称，设置重点
	parser.add_argument("--num_each_class", type=int, default=0)  # 每类个数，一般不再去设置，因为缺陷的生成需要数据量较大
	parser.add_argument("--expansion_ratio","--e", type=int , default = 6)
	parser.add_argument("--device", type=str, default="cuda")  # 注意是在本地还是在远程，本地是cuda或者cuda:0
	parser.add_argument("--ocr", type=str, default='4', help="the number of data batch")
	parser.add_argument("--n_sample_mul", type=int, default = 8)
	parser.add_argument("--n_real_mul",type = int ,default = 2)
	parser.add_argument("--h", type=int, default=64)
	parser.add_argument("--w", type=int, default=64)

	arg = parser.parse_args()
	return arg
def get_parameter_number(model):
	total_num = sum(p.numel() for p in model.parameters())
	trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":
	# masks_finetune64()
	n_feat = 128
	n_classes = 62
	device = "cuda:0"
	n_T = 400
	model = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
				device=device, drop_prob=0.1)  # 注意通道数的改变
	a = get_parameter_number(model) #{'Total': 6610179, 'Trainable': 6610179}
	print(a)
	masks_finetune64()