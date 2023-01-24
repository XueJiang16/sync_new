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
        # feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
        # feature_sim_location=2,  ## No. of conv layer
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
    return x


def norm(features, mean):
    features_mean = features.mean(1).squeeze(0)
    features_norm = features_mean / mean
    features_norm[features_norm > 1] = 1
    features_norm[features_norm < 0] = 0
    features_norm = features_norm.cpu().numpy()
    return features_norm

def feature_sim(feature):
    # print(feature)
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


img_paths = ["/data/csxjiang/val", '/data/csxjiang/ood_data/iNaturalist/images', '/data/csxjiang/ood_data/SUN/images',
            '/data/csxjiang/ood_data/Places/images', '/data/csxjiang/ood_data/Textures/dtd/images_collate']
# img_names = ['ID', 'iNaturalist', 'SUN', 'Places', 'Textures']


#ILSVRC2012_val_00011802.jpg  ILSVRC2012_val_00020514.jpg  ILSVRC2012_val_00020967.jpg  ILSVRC2012_val_00022333.jpg  ILSVRC2012_val_00049692.jpg
# i=0
# img_name = 'ILSVRC2012_val_00049692.JPEG'
# i=4
# img_name = 'braided_0009.jpg'
i = 3
img_name = 'g_grotto_00004308.jpg'
img_path = img_paths[i]
dst_path = "./vis/figure/fig1/"
os.makedirs(dst_path, exist_ok=True)
img = cv2.imread(os.path.join(img_path, img_name))
img1 = img.copy()
img2 = img.copy()
img3 = img.copy()
img4 = img.copy()

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


    # heatmap
    feature_sim1, feature_mean1 = feature_sim(c5)
    feature_sim2, feature_mean2 = feature_sim(c5_th_act)
    # c4_norm = norm(c4, 0.08)
    # c4_th_act_norm = norm(c4_th_act, 0.08)
    c5_norm = norm(c5, 1)
    c5_th_act_norm = norm(c5_th_act, 1)

    c5_norm_mask = cv2.resize(np.uint8(255 * c5_norm), (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    c5_norm_mask = np.float32(c5_norm_mask) / 255
    c5_norm_mask = np.reshape(c5_norm_mask, newshape=-1)
    # plt.hist(c5_norm_mask, density=True, bins=50)
    # plt.savefig("{}_before_hist.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])))
    # plt.close()

    c5_th_act_norm_mask = cv2.resize(np.uint8(255 * c5_th_act_norm), (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
    c5_th_act_norm_mask = np.float32(c5_th_act_norm_mask) / 255
    c5_th_act_norm_mask = np.reshape(c5_th_act_norm_mask, newshape=-1)
    # plt.hist(c5_th_act_norm_mask, density=True, bins=50)
    # plt.savefig("{}_after_hist.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])))
    # plt.close()
    # res1 = show_heatmap(img1, c4_norm)
    # res2 = show_heatmap(img2, c4_th_act_norm)
    res3 = show_heatmap(img3, c5_norm)
    res4 = show_heatmap(img4, c5_th_act_norm)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org = (50, 50)
    fontScale = 0.8
    color = (0, 0, 0)
    thickness = 1
    # res1 = cv2.putText(res1, 'Conf1={}'.format(feature_sim1[0]), (50, 50), font,
    #                    fontScale, color, thickness, cv2.LINE_AA)
    # res3 = cv2.putText(res3, 'Conf2={}'.format(feature_sim2[0]), (50, 50), font,
    #                    fontScale, color, thickness, cv2.LINE_AA)
    # res12 = np.hstack([res1, res2])
    # res34 = np.hstack([res3, res4])
    # res = np.vstack([res12, res34])
    # plt.matshow(C4_features[i])
    cv2.imwrite("{}_before.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])), res3)
    cv2.imwrite("{}_after.jpg".format(os.path.join(dst_path, os.path.splitext(img_name)[0])), res4)
    print("Before score={}, After score={}".format(feature_sim1[0], feature_sim2[0]))



