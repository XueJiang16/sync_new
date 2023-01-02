import torch
import torch.nn as nn
from mmcls.models.builder import build_backbone
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from io import BytesIO
import PIL
import tqdm
import os
import shutil

model_path = "/data/csxjiang/ood_ckpt/pytorch_official/resnet50-19c8e357.pth"

backbone=dict(
            type='ResNetActivation',
            depth=50,
            num_stages=4,
            out_indices=(2, 3,),
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint=model_path)
        )

backbone_th_act = dict(
        type='ResNetActivation',
        depth=50,
        num_stages=4,
        out_indices=(2, 3,),
        style='pytorch',
        th_act_k=0.2,
        th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
        th_act_location=5,  ## No. of conv layer
        feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
        feature_sim_location=2,  ## No. of conv layer
        init_cfg=dict(type='Pretrained', checkpoint=model_path)
)

net = build_backbone(backbone).cuda()
net.init_weights()
net.eval()
net_th_act = build_backbone(backbone_th_act).cuda()
net_th_act.init_weights()
net_th_act.eval()
img_paths = ["/data/csxjiang/val", '/data/csxjiang/ood_data/iNaturalist/images', '/data/csxjiang/ood_data/SUN/images',
            '/data/csxjiang/ood_data/Places/images', '/data/csxjiang/ood_data/Textures/dtd/images_collate']
img_names = ['ID', 'iNaturalist', 'SUN', 'Places', 'Textures']

def oversample(x, group):
    _, c, h, w = x.shape
    x = x[0]
    x = x.reshape((int(c/group), group, -1))
    x = x.mean(0).flatten()
    # x_mean = x.mean(dim=(1,2))
    # value, idx = torch.topk(x_mean, k)
    # x = x[idx[-1]].unsqueeze(0).unsqueeze(0)
    # x = torch.nn.functional.interpolate(x, (128,128),mode="bilinear")
    # x = x.reshape(-1)
    return x

for i in range(len(img_paths)):
    img_path = img_paths[i]
    img_type = img_names[i]
    img_list = os.listdir(img_path)
    dst_path = "./feature_vis_selected/{}".format(img_type)
    os.makedirs(dst_path, exist_ok=True)
    for img_name in tqdm.tqdm(img_list):
        img = cv2.imread(os.path.join(img_path, img_name))
        shutil.copy(os.path.join(img_path, img_name),
                    "{}_orig.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = train_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            c4, c5 = net(img)
            c4_th_act, c5_th_act = net_th_act(img)
            c4 = oversample(c4, 128).cpu().numpy()
            c5 = oversample(c5, 128).cpu().numpy()
            c4_th_act = oversample(c4_th_act, 128).cpu().numpy()
            c5_th_act = oversample(c5_th_act, 128).cpu().numpy()

            ax1 = plt.subplot(221)
            ax1.hist(c4, density=True, bins=100)
            ax2 = plt.subplot(222)
            ax2.hist(c4_th_act, density=True, bins=100)
            ax3 = plt.subplot(223)
            ax3.hist(c5, density=True, bins=100)
            ax4 = plt.subplot(224)
            ax4.hist(c5_th_act, density=True, bins=100)
            plt.savefig("{}.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])))
            plt.close()


