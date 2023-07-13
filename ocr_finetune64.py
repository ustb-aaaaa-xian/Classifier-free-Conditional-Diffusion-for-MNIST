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
from utils.dataloader import dataloader_ocr,dataloader_ocr_color_transform,dataloader_for_G716DVBX
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


def train_data():
    start_time = time.time()
    # hardcoding these here
    # n_epoch = 100
    # batch_size = 8
    # n_T = 400  # 500
    # device = "cuda:0"
    # n_classes = 10
    # n_feat = 128  # 128 ok, 256 better (but slower)
    # lrate = 1e-4
    # save_model = True
    # ocr = "1"
    # dataset_name = "JUZSVEYU"
    # num_each_class = 1
    n_epoch = create_parser().n_epoch
    batch_size = create_parser().batch_size
    n_T = create_parser().n_T
    device = create_parser().device
    n_classes = create_parser().n_classes
    n_feat = create_parser().n_feat
    lrate = create_parser().lrate
    dataset_name = create_parser().dataset_name
    num_each_class = create_parser().num_each_class
    ocr = create_parser().ocr
    n_sample_mul = create_parser().n_sample_mul

    save_dir = f'./Generate_save/Unet64/ocr{ocr}_3_channel/ocr_finetune_{dataset_name}_{num_each_class}_da_/'
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1) # 注意通道数的改变
    # ddpm.load_state_dict(torch.load('./output/diffusion_outputs_ocr_finetune_v1_extreme_data_augment_YQCSSZ18/model_45.pth'))
    ddpm.load_state_dict(torch.load('./data/diffusion_outputs_mnist64_3_channel_/model_70.pth'))
    ddpm.to(device)
    # print(ddpm)
    # 自己写的 ， 加入预训练权重
     # 预训练好的MNIST权重

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64,64)),
        transforms.ToTensor()
        ])  # mnist is already normalised 0 to 1

    if not os.path.exists(save_dir):
        print("save_path not exists,try to mkdirs")
        os.makedirs(save_dir)
    # 尝试多种数据，改写每次的dataset
    dataset = dataloader_ocr_color_transform(data_path = f"./data/ocr{ocr}/{dataset_name}_patch_ocr",transform = tf,num_each_class = num_each_class)
    # dataset = dataloader_ocr_color_transform(data_path = "./data/YQCSSZ18_patch_ocr",transform = tf)
    # dataset = dataloader_for_G716DVBX(data_path = "./data/G716DVBX_patch_ocr" , transform = tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):

        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar: # x是data,c是label;训练/微调
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = (n_epoch-1)/n_epoch * loss_ema + 1/n_epoch * loss.item() # 这个是根据20个epoch来的,所以要更改epoch比例
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        end_time1 = time.time()

        ddpm.eval()
        if (not  ep >= n_epoch -1 )   : # 这句说明最后一次才进行
            continue
        if ep == n_epoch - 1:
            end_time_100_epoch = time.time()
            print(f"after {n_epoch} epochs finetune ,{end_time_100_epoch - start_time} has passed.")
        with torch.no_grad():
            n_sample = n_sample_mul * n_classes
            # 这个for循环是生成以及真实都保存了，再写一个直接生成的函数
            if True:
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 64,64), device, guide_w=w)

                    # append some real images at bottom, order by class also
                    # x_real = torch.Tensor(x_gen.shape).to(device)
                    n_,c_,h_,w_ = x_gen.shape
                    x_real = torch.Tensor(size = (2*n_classes,c_,h_,w_)).to(device)
                    print(x_real.shape)
                    print(c) # 打印最后的标签
                    for k in range(n_classes):
                        # for j in range(int(n_sample / n_classes)):
                        for j in range(int(x_real.shape[0]/n_classes)):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j] # k 被广播机制去与c做等运算，j是行数。
                            except:
                                m = torch.randint(0,n_classes,(1,))
                                idx = (0+m.item()) % len(c)
                            x_real[k + (j * n_classes)] = x[idx]   # 随机展示

                    x_all = torch.cat([x_gen, x_real])
                    # 这个根据数据情况而定
                    # x_all = x_all * -1 + 1 # 此语句将黑底白字的生成效果展示成白底黑字。。且只针对单通道
                    grid = make_grid((x_all-x_all.min())/(x_all.max()-x_all.min()), nrow=10)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                    if  ep > int(n_epoch - 1): # 不进行动图保存
                        # create gif of images evolving over time, based on x_gen_store
                        fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, sharex=True, sharey=True,
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
                                                            vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                            return plots

                        ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store], interval=200, blit=False, repeat=True,
                                            frames=x_gen_store.shape[0])
                        ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                        print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
            # 不记录动图，以及真实图片，只进行生成图片的保存，以及时间的记录.【也可以进行生成图片与真实图片的拼接，查看生成多样性的对比】
            if True:
                # generate_save_dir = f"./Generate_save/Unet64/ocr{ocr}_3_channels/{dataset_name}_{num_each_class}/"
                generate_save_dir = save_dir
                height,width = create_parser().h,create_parser().w # 直接写在parser里
                for w_i, w in enumerate(ws_test): # 如果只想生成 w = 2.0 的话就可以不用for循环
                    generate_s_time = time.time()
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 64,64 ), device, guide_w=w) # 这个sample函数是不是也可以重写，因为根本不需要生成过程的保存，只需要保存生成效果(没必要，生成中间结果的保存并不占用时间)

                    x_ = []
                    for i in range(x_gen.shape[0]):
                        img = x_gen[i]
                        img = transforms.Resize((height,width),interpolation = transforms.InterpolationMode.BICUBIC)(img) # 变成原始长条状
                        j = 0
                        save_path = generate_save_dir+f"image_w{w:.1f}_{i%10}_{j}.png"
                        while os.path.exists(save_path):
                            j += 1
                            save_path = f"{generate_save_dir}image_w{w:.1f}_{i%10}_{j}.png"
                        save_image(img,save_path)
                        print('saved image at ' + save_path,end = "\r")
                    generate_e_time = time.time()
                    print(f"{generate_e_time - generate_s_time}s has passed when generating {n_sample // n_classes} batches of {n_sample} characters of ocr{ocr} model_id {dataset_name},w={w}.")
        end_time2 = time.time()
        # print(f"this epoch has passed {end_time2-start_time}s. forward train has used {end_time1-start_time}s, generating(sampling) has used {end_time2 - end_time1}s.")
        start_time_save_model = time.time()
        if ep == (n_epoch - 1) or ep %100 == 0 and ep != 0: # optionally save model
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")
            end_time_save_model = time.time()
            print(f"{end_time_save_model - start_time_save_model} has passed when saving model.")
def create_parser(): # 这样写是可行的
    parser = argparse.ArgumentParser()

    parser.add_argument("--device",type = str,default = "cuda:1")
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_T", type=int, default=400)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_feat", type=int, default=128)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--ocr", type=str, default='1', help="the number of data batch")
    parser.add_argument("--dataset_name", default = "P0X8CM8Q",type=str)  # 数据集名称，设置重点
    parser.add_argument("--num_each_class", type=int, default=0)  # 每类个数，设置重点

    parser.add_argument("--n_sample_mul",type = int ,default = 4)
    parser.add_argument("--h",type = int ,default = 128)
    parser.add_argument("--w",type = int ,default = 64)


    arg = parser.parse_args()
    return arg




if __name__ == "__main__":
    s_time = time.time()
    train_data()
    end_time = time.time()
    print(f"{end_time-s_time}s has passed after 100 epoches finetune and  {create_parser().n_sample_mul} batches of {create_parser().n_classes} characters[n_sample = {create_parser().n_sample_mul * create_parser().n_classes}] generating.")
