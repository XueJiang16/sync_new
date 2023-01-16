import random

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
import seaborn as sns


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


def oversample(x, group):
    _, c, h, w = x.shape
    x = x[0]
    x = x.reshape((int(c/group), group, -1))
    x = x.mean(1).flatten()
    # x_mean = x.mean(dim=(1,2))
    # value, idx = torch.topk(x_mean, k)
    # x = x[idx[-1]].unsqueeze(0).unsqueeze(0)
    # x = torch.nn.functional.interpolate(x, (128,128),mode="bilinear")
    # x = x.reshape(-1)
    return x


def norm(features, mean):
    features_mean = features.mean(1).squeeze(0)
    features_norm = features_mean / mean
    features_norm[features_norm > 1] = 1
    features_norm[features_norm < 0] = 0
    features_norm = features_norm.cpu().numpy()
    return features_norm

def feature_sim(feature):
    feature_crops = feature.flatten(-2)
    patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
    patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
    return patch_sim, patch_mean

def show_heatmap(img: np.ndarray,
                 mask: np.ndarray,
                 use_rgb: bool = False,
                 colormap: int = cv2.COLORMAP_JET,
                 image_weight: float = 0.5):
    mask = cv2.resize(np.uint8(255 * mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(mask, colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        img = np.float32(img) / 255
    else:
        img = np.float32(img)

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


img_path = "/data/csxjiang/val/"
img_name = 'ILSVRC2012_val_00000837.JPEG'
dst_path = "./vis_res/vis_heatmap/"
os.makedirs(dst_path, exist_ok=True)
img = cv2.imread(os.path.join(img_path, img_name))
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
    # c4_th_act, c5_th_act = net_th_act(img)
    c4 = c4.mean(1).squeeze(0).cpu().numpy()
    sns.heatmap(c4, cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True))
    plt.savefig("c4_{}.jpg".format(os.path.join(dst_path, img_name)))
    plt.close()
