from mmcv.runner import BaseModule    # noqa
import torch  # noqa
import torch.nn as nn

import os
import numpy as np  # noqa
from collections import Counter  # noqa

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa


def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")


@OOD.register_module()
class PatchSim(BaseModule):
    def __init__(self, num_crop, img_size, threshold, order=1, ood_detector=None, mode='cosine',**kwargs):
        super(PatchSim, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order
        self.mode = mode

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            img = input['img']
            img_size = self.img_size
            crop_size = int(img_size / self.num_crop)
            corner_list = []
            crops = []
            for h in range(self.num_crop):
                for w in range(self.num_crop):
                    corner_list.append([h * crop_size, w * crop_size])
            for h, w in corner_list:
                crop = img[:, :, h: h + crop_size, w: w + crop_size]
                input['img'] = crop
                _, crop_feature = self.ood_detector.classifier(return_loss=False, softmax=False,
                                                               post_process=False, require_features=True, **input)
                crops.append(crop_feature)
            input['img'] = img
            input['type'] = type
            patch_sim = 0
            count = 0
            for i in range(len(crops)-1):
                for j in range(i+1, len(crops)):
                    if self.mode == 'cosine':
                        tmp = - (crops[i] * crops[j]).sum(dim=1)
                        tmp = tmp / (torch.norm(crops[i], dim=1) * torch.norm(crops[j], dim=1))
                        tmp = (tmp + 1) / 2
                    elif self.mode == 'euclidean':
                        tmp = torch.norm(crops[i]-crops[j], dim=1)
                    patch_sim += tmp
                    count += 1
            patch_sim /= count
            # ood_scores = patch_sim
            if self.has_ood_detector:
                ood_scores, _ = self.ood_detector(**input)
                patch_sim = ((1 / self.threshold) ** (self.order)) * torch.pow(patch_sim, self.order)
                patch_sim[patch_sim > 1] = 1
                ood_scores *= patch_sim
            else:
                ood_scores = patch_sim
        return ood_scores, type


@OOD.register_module()
class FeatureMapSim(BaseModule):
    def __init__(self, num_crop, img_size, threshold, fuse_const=1,k=1, order=1, ood_detector=None, mode='cosine',**kwargs):
        super(FeatureMapSim, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order
        self.mode = mode
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.k = k
        self.fuse_const=fuse_const


    def kap(self, x):
        b, c, h, w = x.shape
        x_gap = self.gap(x).view(b, c)

        ## kap
        x = x.view(b, c, -1)
        ## h*w -> top k
        num = int(self.k * (h * w))
        # num = int(self.k * h)
        topk_v, _ = x.topk(num, dim=-1)
        out = topk_v.mean(dim=-1)
        # topk_v, _ = out.topk(num, dim=-1)
        # out = topk_v.mean(dim=-1)
        mean_gap = x_gap.mean(dim=-1)
        mean_kap = out.mean(dim=-1)
        out = out * (mean_gap / mean_kap).unsqueeze(-1)
        return out

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            _, feature_c5 = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
                                                         require_backbone_features=True, **input)
            print(feature_c5)
            assert False
            ##########
            # _, features_orig = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
            #                                                 th_act=False, require_features=True, **input)
            # _, features_th_act = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
            #                                                   th_act=True, require_features=True, **input)
            # kl_sim = -torch.nn.functional.kl_div(features_orig, features_th_act, reduction='none').mean(1)

            input['type'] = type
            if self.mode in ['cosine', 'euclidean']:
                feature_crops = torch.nn.functional.interpolate(feature_c5, size=self.num_crop, mode='bilinear')
                feature_crops = feature_crops.flatten(2)
                patch_sim = 0
                count = 0
                for i in range(self.num_crop**2-1):
                    for j in range(i+1, self.num_crop**2):
                        if self.mode == 'cosine':
                            tmp = - (feature_crops[:, :, i] * feature_crops[:, :, j]).sum(dim=1)
                            tmp = tmp / (torch.norm(feature_crops[:, :, i], dim=1) *
                                         torch.norm(feature_crops[:, :, j], dim=1))
                            tmp = (tmp + 1) / 2
                        elif self.mode == 'euclidean':
                            tmp = torch.norm(feature_crops[:, :, i]-feature_crops[:, :, j], dim=1)
                        patch_sim += tmp
                        count += 1
                patch_sim /= count
                # ood_scores = patch_sim
            elif self.mode == 'std':
                # feature_c5 = feature_c5[:,:,1:6,1:6]
                feature_crops = feature_c5.flatten(2)  # (B, C, H*W)
                patch_sim = feature_crops.std(-1).mean(-1)
            elif self.mode == 'kap':
                patch_mean = self.kap(feature_c5).unsqueeze(-1)
                patch_sim = torch.abs(feature_c5.flatten(2) - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)

            elif self.mode == 'mean':
                # feature_c5 = feature_c5[:,:,1:6,1:6]
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
                # c5_sum = feature_c5.sum(dim=[1,2,3])
                # ratio = (c5_sum - 20000) / 10000
                # patch_sim = ratio * patch_sim
                ## exclude 1-sigma
                # patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=-1)
                # patch_sim_std = patch_sim.std(dim=-1)
                # patch_sim_mean = patch_sim.mean(dim=-1)
                # lower = patch_sim_mean-patch_sim_std
                # upper = patch_sim_mean+patch_sim_std
                # lower = lower.unsqueeze(1)
                # upper = upper.unsqueeze(1)
                # patch_sim = torch.where(patch_sim < lower, lower, patch_sim)
                # patch_sim = torch.where(patch_sim > upper, upper, patch_sim)
                # patch_sim = patch_sim.mean(-1)
            elif self.mode == 'median':
                feature_crops = feature_c5.flatten(2)
                patch_median = feature_crops.median(-1)[0].unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_median).flatten(1).median(dim=-1)[0] # for ID: .mean(dim=-2)

            elif self.mode == 'channel_mean':
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(1).unsqueeze(1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'fuse_mean':
                feature_crops = feature_c5.flatten(2)
                channel_mean = feature_crops.mean(1).unsqueeze(1)  # (N, C, H*W) -> (N, 1, H*W)
                channel_sim = torch.abs(feature_crops - channel_mean).mean(dim=(-1, -2))
                spatial_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C, 1)
                spatial_sim = torch.abs(feature_crops - spatial_mean).mean(dim=(-1, -2))
                patch_sim = channel_sim + spatial_sim
            elif self.mode == 'extract_feature_sim':
                # (N, C, H*W) -> (N, C) -> (C,) -> argsort -> topK_idx -> id_ood_inference -> feature_crops[:, topK_idx]
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C, 1)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(0, 2))  # for ID: .mean(dim=(0, 2))
            elif self.mode is None:
                patch_sim=0
            ood_scores = patch_sim
        if self.has_ood_detector:
            ood_scores, _ = self.ood_detector(**input)
            # patch_sim = ((1 / self.threshold) ** (self.order)) * torch.pow(patch_sim, self.order)
            # patch_sim[patch_sim > 1] = 1
            # with torch.no_grad():

            # ood_scores = ood_scores * kl_sim * patch_sim
            ood_scores *= self.fuse_const
            ood_scores += patch_sim
            # print("mean:", ood_scores.mean())
            # print("std:", ood_scores.std())
            # exit()
            ## add strategies
            # batch_ratio = (ood_scores / patch_sim).abs().cpu()
            # for i in range(batch_ratio.shape[0]):
            #     ratio = batch_ratio[i].data
            #     if ratio > 1:
            #         ratio = str(int(ratio))
            #         ratio = 10**(len(ratio)-1)
            #     else:
            #         ratio = str(int(1/ratio))
            #         ratio = 10 ** (1-len(ratio))
            #     ood_scores[i] += (patch_sim[i] * ratio)
        else:
            ood_scores = patch_sim
        return ood_scores, type
