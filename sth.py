import torch
import torch.nn as nn
import sys
import os
import numpy as np
import cv2
import argparse
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

# 对下载好的cifar数据集进行一定程度的可视化
def unpickle(file):
    import pickle
    with open("./data/CIFAR10/cifar-10-batches-py/" + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def img_save(datas,labels,label_names,k:int  = 100):
    save_dir = "./data/cifar_visualiaed"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(k):
        data = datas[i]
        label = labels[i]
        label_name = label_names[int(label)]
        img = data.reshape(-1,1024)
        img_ = np.zeros(shape = (32,32,3))
        r = img[0,:].reshape(1,32,32)
        g = img[1,:].reshape(1,32,32)
        b = img[2,:].reshape(1,32,32)
        img_[:,:,0] = r
        img_[:,:,1] = g
        img_[:,:,2] = b

        # 此处归一化操作主要是为了模拟transforms里的ToTensor（）操作，查看归一化后的可视化图像是什么样子。
        img_ = img_ - img_.min()/(img_.max()-img_.min())
        i = 0
        img_ = cv2.resize(img_,(256,256),interpolation=cv2.INTER_CUBIC)
        save_path = os.path.join(save_dir, f"{label_name}_{i}.png")
        while os.path.exists(save_path):
            i += 1
            save_path = os.path.join(save_dir,f"{label_name}_{i}.png")

        cv2.imwrite(save_path,img_)

# 一个测试Tensor保存成图像的操作
def test_img_save():
    # 4 / save_image的自己测试操作 ，其将Tensor转化成可以保存的图像
    # x = torch.rand((40, 3, 64, 64))
    x = torch.arange(0,255,1) # 由黑到白的渐变
    x = torch.unsqueeze(x,dim = 0)
    x = x.repeat(x.shape[1],1)
    x = torch.unsqueeze(x,dim = 0).repeat(3,1,1)
    print(x.shape)
    # 0-1的数据* 255 加上0.5【应该是为了产生有255的数据】，在0-5截断，变成uint8，变换维度（将通道维度放到最后），转成numpy
    # x_1 = x.mul(255).add_(0.5).clamp(0, 255).type(torch.uint8).permute(0, 2, 3, 1).numpy()
    x_1 = x.clamp(0, 255).type(torch.uint8).permute(1,2,0).numpy()

    # x_1 = x[0]  # 取出第一个数据
    from PIL import Image
    im_x1 = Image.fromarray(x_1)  # 将ndarray转成PIL.Image.Image image格式
    file_path = "test.jpg"
    im_x1.save(file_path)  # 进行保存


# 加载预训练模型
def load_pretrain_pth():
    # 1 / load pre_train pth
    pt_path = "logs/training.pt"
    saveed_model_pt = torch.load(pt_path)
    # print(saveed_model_pt)
    print(len(saveed_model_pt),saveed_model_pt[0].shape,saveed_model_pt[1].shape)
    a_ = saveed_model_pt[0].detach().numpy()
    print(a_.shape)

    x = torch.rand((4,256,7,7))
    avg = nn.AvgPool2d(kernel_size=7,stride=7,padding =0)
    out = avg(x)
    print(out.shape)

# test ddpm_schedules函数
def test_ddpm_():
    # 2 / test ddpm_schedules
    beta1 = torch.tensor(1e-4)
    beta2 = torch.tensor(0.02)
    T = torch.tensor(500)
    dic = ddpm_schedules(beta1,beta2,T)
    for k,v in dic.items():
        print(k,v.shape)

# visualize cifar dataset and save
def visulize_dataset():
    # 3 / visualize  an CIFAR10 IMAGE

    # 打开cifar-10数据集文件目录
    data_batch = unpickle("data_batch_1")
    print(type(data_batch))  # <class 'dict'>
    for k, v in data_batch.items():
        print(k, end=" ")  # b'batch_label' b'labels' b'data' b'filenames'
    cifar_data = data_batch[b'data']
    cifar_label = data_batch[b'labels']  # 两个都是list
    cifar_data, cifar_label = np.array(cifar_data), np.array(cifar_label)
    print(cifar_data.shape, cifar_label.shape)  # (10000,3072) (10000,)
    label_name = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog',
                  'frog', 'horse', 'ship', 'truck']
    img_save(cifar_data,cifar_label,label_name,k = 100)# 一个函数用来可视化cifar10图像并进行可视化后的保存

# 统计图片大小信息，包括min（），max（）
def count_if_small(path):
    heights = list()
    widths = list()
    mask_dir = path
    for file in os.listdir(mask_dir):
        mask_file = os.path.join(mask_dir,file)
        img = cv2.imread(mask_file)
        height,width,channel = img.shape

        heights.append(height)
        widths.append(width)
        if height <64 or width <64:
            print(file)
        else:
            continue
    heights = np.array(heights)
    widths = np.array(widths)
    print(heights.min(),heights.max(),heights.mean(),heights.std()) # 27 974 , mean:129,最小的是27，是不是resize的时候有问题
    print(widths.min(), widths.max(), widths.mean(), widths.std()) # 28 1320 , mean:199
    return heights,widths

# 查看预训练好的模型
def load_model():
    from scrip import DDPM,ContextUnet
    n_T = 400
    device = "cuda"
    ddpm = DDPM(nn_model = ContextUnet(in_channels=3,n_feat = 128,n_classes = 10),betas = (1e-4,0.02),n_T = n_T,device = device,drop_prob = 0.1)
    ddpm.to(device)
    m = torch.load('./logs/mnist_3_channel_model_49.pth',map_location=device)
    print(type(m)) # class collections.OrderedDict
    for k,v in m.items():
        print(f"{k}: {v.shape}")
    ddpm.load_state_dict(torch.load('./logs/mnist_3_channel_model_49.pth',map_location=device)) #这个模型是部署在两块卡上的。。。
    # print(ddpm)

# parser解析器的使用
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_T", type=int, default=400)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_feat", type=int, default=128)
    parser.add_argument("--lrate", type=float, default=1e-4)
    parser.add_argument("--dataset_name", type=str)  # 数据集名称，设置重点
    parser.add_argument("--num_each_class", type=int, default=3)  # 每类个数，设置重点
    parser.add_argument("--save_dir", type=str, default="./output/")

    # 这句会出问题，save_dir写在train_data函数里
    # parser.add_argument("--save_dir" ,type = str ,default = f'./output/ocr3_3_channel/ocr_finetune_{parser.dataset_name}_{parser.num_each_class}_da_/',help= "saved directory")

    arg = parser.parse_args()
    print(type(arg))
    print(arg.lrate, arg.num_each_class, arg.save_dir)
    return arg
if __name__ == "__main__":
    # load_model()
    test_img_save()
    pass

















