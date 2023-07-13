"""
7.6
    每个模型的导入都得import对应的DDPM以及ContextUnet
    1)generate_ocr_paticular()是对ocr_byclass微调特定类别的生成
    2)generate_mask_paticular()是对mask微调完成的模型特定类别的生成
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.datasets import CelebA
from utils.dataloader import dataloader_v1
"""
    直接用权值文件去进行生成图片以及保存
"""
from ocr_finetune import ContextUnet,DDPM
def generate_save_cifar10_64():
    n_epoch = 10
    batch_size = 16
    n_T = 400  # 500
    device = "cuda:0"
    n_classes = 10  # no n_classes
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True

    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance

    img_save_dir = "./data/Generate_save/CIFAR10_64"
    if not os.path.exists(img_save_dir):
        print("save_dir not exists,be created!")
        os.makedirs(img_save_dir)
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    # 模型选择
    ddpm.load_state_dict(torch.load('./data/diffusion_output_Unet64_cifar10_conditional_255/model_39.pth'))
    ddpm.to(device)
    for ep in range(n_epoch):
        ddpm.eval()
        with torch.no_grad():
            for w_i,w in enumerate(ws_test):

                x_gen,x_gen_store = ddpm.sample(n_sample = 40,size= (3,64,64),device = device ,guide_w = w)
                # 这个函数返回类型需要仔细看 x_gen是Tensor,但是x_gen_store是ndarray，其将Tensor转成numpy之后又进行的列表拼接又转成ndarray

                # 把Tensor保存成PIL
                n = x_gen.shape[0]
                for i in range(n):

                    img = x_gen[i]
                    img = img.cpu()
                    img_Tensor = torch.Tensor(img)
                    img_Tensor = (img_Tensor - img_Tensor.min())/(img_Tensor.max()-img_Tensor.min())
                    save_image(img_Tensor,os.path.join(img_save_dir,f'generate_cifar10_64_ep{ep}_w{w}_{i}.jpg'))

def generate_save_ocr_finetune_extreme():
    time1 = time.time() # 为了记录载入模型的时间
    # n_epoch = 2
    # batch_size = 16
    # n_T = 400  # 500
    # device = "cuda:0"
    # n_classes = 10  # no n_classes
    # n_feat = 128  # 128 ok, 256 better (but slower)
    arg = create_parser()
    n_epoch = arg.n_epoch
    n_T = arg.n_T
    device = arg.device
    n_feat = arg.n_feat
    n_classes = arg.n_classes
    dataset_name = arg.dataset_name
    ocr = arg.ocr
    num_each_class = arg.num_each_class
    height = arg.h
    width = arg.w
    ws_test = [0.0, 0.5, 2.0]  #   strength of generative guidance

    # save_dir保存路径 以及 load_state_dict 模型的选择
    # save_dir = "./data/Generate_save/ocr3/ocr_3_channels/HDWG557D_3/"
    # 生成保存直接在另一个目录Generate_save
    save_dir = f"./Generate_save/ocr{ocr}_3_channels/{dataset_name}_{num_each_class}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    model_path = f"./output/ocr{ocr}_3_channel/ocr_finetune_{dataset_name}_{num_each_class}_da_/model_99.pth"
    ddpm.load_state_dict(torch.load(model_path))
    ddpm.to(device)
    time2 = time.time()
    print(f"load_model has passed {time2-time1}s.")
    for ep in range(n_epoch):
        ddpm.eval()
        with torch.no_grad():
            # n_sample = 4 * n_classes
            """
                记录一套数据10个字符的生成时间
            """

            n_sample = n_classes
            # 保存多个图 / 保存单张图
            for w_i, w in enumerate(ws_test):
                s_t = time.time()
                x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28,28 ), device, guide_w=w) # 这个sample函数是不是也可以重写，因为根本不需要生成过程的保存，只需要保存生成效果

                # 将一些Tensor还原成原始形状，是不是可以
                # append some real images at bottom, order by class also
                x_ = []
                # j = 0
                for i in range(x_gen.shape[0]):
                    # s_t = time.time()
                    img = x_gen[i]
                    # img = img * -1 + 1  # 在这进行反转生成【生成白底黑字的】
                    img = transforms.Resize((height,width),interpolation = transforms.InterpolationMode.BICUBIC)(img) # 变成原始长条状
                    # x_.append(img)
                    j = 0
                    save_path = save_dir+f"image_w{w:.1f}_{i%10}_{j}.png"
                    # save_path = f"{save_dir}image_w{w:.2f}_{i}_{j}.png"
                    while os.path.exists(save_path):
                        j += 1
                        # save_path = save_dir+f"image_w{w:%.2f}_{i}_{j}.png"
                        save_path = f"{save_dir}image_w{w:.1f}_{i%10}_{j}.png"
                    save_image(img,save_path)

                # grid = make_grid(x_ * -1 + 1, nrow=10)
                # grid = make_grid(x_, nrow=10)
                # save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_path,end = "\r")
                e_t = time.time()
                print(f"{e_t-s_t}s has passed when generating a batch of 10 characters of ocr{ocr},model_id{dataset_name}.")
                with open("time_record","w") as f:
                    print(f"{e_t-s_t}s has passed when generating a batch of 10 characters of ocr{ocr},model_id{dataset_name}.",file = f)

def generate_save_ocr_finetune64():
    time1 = time.time() # 为了记录载入模型的时间
    n_epoch = 2
    batch_size = 16
    n_T = 400  # 500
    device = "cuda:0"
    n_classes = 10  # no n_classes
    n_feat = 128  # 128 ok, 256 better (but slower)
    height = 50
    width = 25
    ocr = "1"
    dataset_name = ""
    num_each_class = 1

    ws_test = [0.0, 0.5, 2.0]  #   strength of generative guidance

    # save_dir保存路径 以及 load_state_dict 模型的选择
    # save_dir = "./data/Generate_save/ocr3/ocr_3_channels/HDWG557D_3/"
    # 生成保存直接在另一个目录Generate_save
    save_dir = f"./Generate_save/ocr{ocr}_3_channels/{dataset_name}_{num_each_class}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    model_path = f"./output/ocr{ocr}_3_channel/ocr_finetune_{dataset_name}_{num_each_class}_da_/model_99.pth"
    ddpm.load_state_dict(torch.load(model_path))
    ddpm.to(device)
    time2 = time.time()
    print(f"load_model has passed {time2-time1}s.")
    for ep in range(n_epoch):
        ddpm.eval()
        with torch.no_grad():
            # n_sample = 4 * n_classes
            """
                记录一套数据10个字符的生成时间
            """
            n_sample = n_classes
            # 保存多个图 / 保存单张图
            for w_i, w in enumerate(ws_test):
                s_t = time.time()
                x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28,28 ), device, guide_w=w) # 这个sample函数是不是也可以重写，因为根本不需要生成过程的保存，只需要保存生成效果

                # 将一些Tensor还原成原始形状，是不是可以
                # append some real images at bottom, order by class also
                x_ = []
                # j = 0
                for i in range(x_gen.shape[0]):
                    # s_t = time.time()
                    img = x_gen[i]
                    # img = img * -1 + 1  # 在这进行反转生成【生成白底黑字的】
                    img = transforms.Resize((height,width),interpolation = transforms.InterpolationMode.BICUBIC)(img) # 变成原始长条状
                    # x_.append(img)
                    j = 0
                    save_path = save_dir+f"image_w{w:.1f}_{i%10}_{j}.png"
                    # save_path = f"{save_dir}image_w{w:.2f}_{i}_{j}.png"
                    while os.path.exists(save_path):
                        j += 1
                        # save_path = save_dir+f"image_w{w:%.2f}_{i}_{j}.png"
                        save_path = f"{save_dir}image_w{w:.1f}_{i%10}_{j}.png"
                    save_image(img,save_path)

                # grid = make_grid(x_ * -1 + 1, nrow=10)
                # grid = make_grid(x_, nrow=10)
                # save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_path,end = "\r")
                e_t = time.time()
                print(f"{e_t-s_t}s has passed when generating a batch of 10 characters of ocr{ocr},model_id{dataset_name}.")
                with open("time_record","w") as f:
                    print(f"{e_t-s_t}s has passed when generating a batch of 10 characters of ocr{ocr},model_id{dataset_name}.",file = f)

"""
    此函数为了生成特定的某个标签的大量样本。
    需要额外的sample函数
"""
def generate_ocr_paticular():
    time1 = time.time()
    from pretrain_emnist_byclass_finetune import DDPM

    n_feat = 128
    n_classes = 62
    n_T = 400
    device = "cuda:0"
    n_sample = 50
    ocr = 4
    c = 1 # 生成的标签数
    dataset_name = "QJCXDX0Y"
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(f"./Generate_save/ocr_byclass_finetune/ocr4/{dataset_name}_8/{dataset_name}_model_49.pth"))
    ddpm.to(device)
    save_dir = f"./Generate_save/ocr{ocr}/specific_class/{dataset_name}_c_{c}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    generate_save_dir = save_dir
    height, width = 50,30
    w = 2.0
    time2 = time.time()
    print(f"loading model has passed {time2 - time1}s.")
    generate_s_time = time.time()
    ddpm.eval()
    with torch.no_grad():
        x_gen, x_gen_store = ddpm.sample_for_paticular_c(n_sample = n_sample,size = (3,28,28),device = device,guide_w = w,
                                                   c = 1)

        # x_gen = (x_gen - x_gen.min())/(x_gen.max() - x_gen.min())
        print("generating done !")
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
            save_path = generate_save_dir + f"{dataset_name}_image_w{w:.1f}_{int(c)}_{j}.png"
            while  os.path.exists(save_path):
                j += 1
                save_path = generate_save_dir + f"{dataset_name}_image_w{w:.1f}_{int(c)}_{j}.png"
            img = (img - img.min()) / (img.max() - img.min())
            save_image(img, save_path)
            print('saved image at ' + save_path, end="\r")
        generate_e_time = time.time()
    print(
        f"{generate_e_time - generate_s_time}s has passed when generating and saving {n_sample } characters.")

def generate_mask_paticular():

    time1 = time.time()
    from masks_finetune64 import DDPM
    from masks_finetune64 import ContextUnet
    arg = create_parser()
    n_feat = 128
    n_classes = 10
    n_T = 400
    device = "cuda:0"
    n_sample = arg.n_sample
    c = arg.c # 生成的标签类别
    dataset_name = "T6J9BF9K"
    clamp_right = arg.clamp_right
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(f"./output/mask/masks_{dataset_name}_cifar64_padding_200/{dataset_name}_model_199.pth",map_location="cuda:0"))
    ddpm.to(device)
    # save_dir = f"./Generate_save/mask/{dataset_name}_2/{dataset_name}_c_{c}/" # 目录命名规范一点
    save_dir = f"./Generate_save/mask/{dataset_name}_2_Normalize/{dataset_name}_c_{c}/" # 目录命名规范一点
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    generate_save_dir = save_dir
    height, width = arg.h,arg.w
    w = 2.0
    time2 = time.time()
    print(f"loading model has passed {time2 - time1}s.")
    generate_s_time = time.time()
    ddpm.eval()
    with torch.no_grad():
        x_gen, x_gen_store = ddpm.sample_for_paticular_c(n_sample = n_sample,size = (3,64,64),device = device,guide_w = w,
                                                   c = c)

        print("generating done !")
        for i in range(x_gen.shape[0]):
            img = x_gen[i]
            print(f"generating original,img.min(),img.max():{img.cpu().numpy().min(),img.cpu().numpy().max()}")
            # 生成图片亮度不够以及很黑，进行截断。
            # img = torch.clamp(img,-1,clamp_right)
            print(f"after clamp,        img.min(),img.max():{img.cpu().numpy().min(),img.cpu().numpy().max()}")
            img = (img - img.min())/(img.max() - img.min())
            # print(f"after normalize,    img.min(),img.max():{img.cpu().numpy().min(),img.cpu().numpy().max()}")
            if height < 28 or width < 28:
                img = transforms.Resize((height, width),
                                        interpolation=transforms.InterpolationMode.NEAREST)(
                    img)
            else:
                img = transforms.Resize((height, width),
                                        interpolation=transforms.InterpolationMode.BICUBIC)(
                    img)  # 变成原始长条状
            # Normalize回原始的
            print(f"img_mean,std:{img.mean(),img.std()}")
            if c == 0:
                original_mean,original_std = 0.82,0.21
            elif c == 1:
                original_mean, original_std = 0.95, 0.097
            elif c == 2:
                original_mean, original_std = 0.886,0.146
            # tf_normalize = transforms.Normalize([original_mean,original_mean,original_mean],[original_std,original_std,original_std])
            # img = tf_normalize(img)
            j = 0
            print(f"img.mean(),img.std(),img.min(),img.max():{img.cpu().mean(),img.cpu().std(),img.cpu().numpy().min(),img.cpu().numpy().max()}")
            img = (img - img.min())/(img.max() - img.min())
            print(
                f"after 0-1 normalize,img.mean(),img.std(),img.min(),img.max():{img.cpu().mean(), img.cpu().std(), img.cpu().numpy().min(), img.cpu().numpy().max()}")
            save_path = generate_save_dir + f"{dataset_name}_image_{int(c)}_clamp_right{clamp_right}_{j}.png"
            while  os.path.exists(save_path):
                j += 1
                save_path = generate_save_dir + f"{dataset_name}_image_{int(c)}_clamp_right{clamp_right}_{j}.png"
            # img = (img - img.min()) / (img.max() - img.min())
            save_image(img, save_path)
            print('saved image at ' + save_path, end="\r")
        generate_e_time = time.time()
    print(
        f"{generate_e_time - generate_s_time}s has passed when generating and saving {n_sample } characters.")

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epoch",type = int ,default = 1)
    parser.add_argument("--batch_size",type = int ,default = 64)
    parser.add_argument("--n_T",type = int ,default = 400)
    parser.add_argument("--device",type = str ,default = "cuda:0")
    parser.add_argument("--n_classes" ,type = int ,default = 10)
    parser.add_argument("--n_feat" ,type =int ,default = 128)
    parser.add_argument("--dataset_name" , default = "S1ONVPHW",type =str)#
    parser.add_argument("--ocr",type = str,default = 1,help = "the ocr type of 1/2/3")#
    parser.add_argument("--h",type = int ,default = 64 ,help = "the height of original height mean")#
    parser.add_argument("--w",type = int ,default = 64,help = "the width of original width mean")#
    parser.add_argument("--num_each_class",type = int ,default =  1)#
    parser.add_argument("--c",type = int ,default = 0 )
    parser.add_argument("--n_sample",type = int ,default = 20)
    parser.add_argument("--clamp_right",type = float,default = 1.)
    # parser.add_argument("--lrate" ,type = float,default = 1e-4)

    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    generate_mask_paticular()


