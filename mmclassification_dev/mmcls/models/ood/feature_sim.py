from mmcv.runner import BaseModule    # noqa
import torch  # noqa
import torch.nn as nn
import time
import warnings

import os
import cv2
import numpy as np  # noqa
from collections import Counter  # noqa

#for gmm
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM



from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa


def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")

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


@OOD.register_module()
class FeatureReweight(BaseModule):
    def __init__(self, ood_detector=None, mode='mean',**kwargs):
        super(FeatureReweight, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.mode = mode

    def gmm_score(self, x, out_dir='./'):
        x = x.cpu().detach().numpy()
        x = x.reshape(-1, 1)
        gmm = GMM(n_components=2, max_iter=100, random_state=10, covariance_type='full')
        # find useful parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = gmm.fit(x)
        mean = model.means_
        covs = model.covariances_
        weights = model.weights_

        # print('Dis1 mean={}, std={}, weight={}'.
        #       format(float(mean[0][0]), np.sqrt(float(covs[0][0][0])), weights[0]))
        # print('Dis2 mean={}, std={}, weight={}'.
        #       format(float(mean[1][0]), np.sqrt(float(covs[1][0][0])), weights[1]))

        # create necessary things to plot
        # x_axis = np.arange(-0.1, 1.1, 0.001)
        # y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0]))) * weights[0]  # 1st gaussian
        # y_axis1 = norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0]))) * weights[1]  # 2nd gaussian
        # plt.hist(x, density=True, color='black', bins=60)
        # plt.plot(x_axis, y_axis0, label='Dis 1')
        # plt.plot(x_axis, y_axis1, label='Dis 2')
        # plt.plot(x_axis, y_axis0 + y_axis1, ls='dashed', label='Mixed Dis')
        # plt.xlim(-0.1, 1.1)
        # # plt.ylim(0.0, 2.0)
        # plt.xlabel(r"X")
        # plt.ylabel(r"Density")
        # plt.legend()
        # filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        # plt.savefig(os.path.join(out_dir, '{}.jpg'.format(filename)))
        # plt.close('all')

        if mean[0][0] > mean[1][0]:
            forg_idx = 0
            bacg_idx = 1
        else:
            forg_idx = 1
            bacg_idx = 0
        forg_mean = mean[forg_idx][0]
        forg_std = np.sqrt(float(covs[forg_idx][0][0]))
        forg_weight = weights[forg_idx]
        bacg_mean = mean[bacg_idx][0]
        bacg_std = np.sqrt(float(covs[bacg_idx][0][0]))
        bacg_weight = weights[bacg_idx]

        single_score = (forg_mean - bacg_mean) * forg_weight / bacg_weight
        # x_mean = x.mean()
        # gmm_mean = forg_weight * forg_mean + bacg_weight * bacg_mean
        # single_score = np.abs(forg_mean - gmm_mean) * forg_weight
        # + np.abs(bacg_mean - gmm_mean) * bacg_weight
        return single_score



    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']

        with torch.no_grad():
            _, feature_c5 = self.ood_detector.classifier(return_loss=False, softmax=False, post_process=False,
                                                         require_backbone_features=True, **input)
            input['type'] = type
            if self.mode == 'mean':
                feature_crops = feature_c5.flatten(2)
                # feature_crops = feature_crops[:,::4].contiguous()
                # value, index = feature_crops.mean(-1).max(dim=-1)  # (N, C, H*W) -> (N, C)
                # patch_sim = torch.zeros_like(value)
                # for i,j in enumerate(index):
                #     patch_sim[i] = torch.abs(feature_crops[i,j] - value[i]).mean(dim=-1)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
                # patch_sim = torch.clamp((feature_crops - patch_mean), min=0).mean(dim=(-1, -2))
                # feature_crops = feature_crops-patch_mean
                # feature_crops[feature_crops < 0] = 0
                # patch_sim = feature_crops.mean(dim=(-1,-2))
            # elif self.mode == 'cosine':
            #     feature_target = torch.ones_like(feature_c5)
            #     patch_sim =
            elif self.mode == 'clamp':
                feature_crops = feature_c5.flatten(2)
                feature_crops = torch.clamp(feature_crops, min=0, max=1.2)
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                # patch_mean = (0.4 / (feature_crops.std(dim=-1)+0.05)).unsqueeze(-1)
                patch_sim = torch.clamp((feature_crops - patch_mean), min=0, max=0.5).mean(
                    dim=(-1, -2))  # for ID: .mean(dim=-2)
                # patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'gmm':
                filenames = [x['filename'] for x in input['img_metas']]
                if 'ood_data' in filenames[0]:
                    mid_path = 'OOD'
                else:
                    mid_path = 'ID'
                out_dir = os.path.join('./vis_gmm/', mid_path)
                os.makedirs(out_dir, exist_ok=True)
                feature_crops = feature_c5.flatten(2)
                batch_size, channel, _ = feature_crops.shape
                sub_channel = 64
                score = []
                for i in range(batch_size):
                    x = feature_crops[i]
                    # val, idx = torch.topk(x.mean(dim=-1), k=2048)
                    # x = x[idx].flatten()
                    x = x.flatten()
                    x = x[::64].contiguous()
                    # x = x.reshape(int(channel/sub_channel), sub_channel, -1)
                    # x = x.mean(0)
                    # single_score = np.mean(list(map(self.gmm_score, x)))
                    single_score = self.gmm_score(x)
                    score.append(single_score)
                patch_sim = torch.tensor(score).to("cuda:{}".format(self.local_rank))

            if self.mode == 'vit':
                print(self.ood_detector.classifier.backbone.layers[11])
                target_layer = self.ood_detector.classifier.backbone.layers[11]
                ln = target_layer.ln1
                attn = target_layer.attn.qkv.weight
                print(attn.shape)
                x = ln(x)
                assert False
                x = feature_c5
                B, _, C = x.shape

                assert False
                # feature_c5 (B, 768, 24, 24)
                feature_tokens = feature_c5.flatten(2)  # (B, 768, 576)
                feature_tokens = feature_tokens.permute((0, 2, 1))  # (B, 576, 768)
                feature_tokens_ = feature_tokens / feature_tokens.norm(dim=-1).unsqueeze(-1)  # (B, 576, 768N)
                # cls_token, patch_token = feature_tokens_[:, :, 0].unsqueeze(-1), feature_tokens_[:, :, 1:]
                # feature_affinity = torch.einsum("bdi,bdj->bij", cls_token, patch_token)
                feature_affinity = torch.einsum("bid,bjd->bij", feature_tokens_, feature_tokens_)  # (B, 576, 576)
                # filenames = [x['filename'] for x in input['img_metas']]
                # output_path = "./feature_affinity_vis"
                # os.makedirs(output_path, exist_ok=True)
                # for i in range(len(feature_tokens)):
                #     img_name = filenames[i]
                #     img = cv2.imread(img_name)
                #     # f = feature_affinity[i, 288].reshape((24, 24)).cpu()
                #     f = feature_affinity[i, 0].reshape((24, 24)).cpu()
                #     f = (f + 1) / 2
                #     cam = show_heatmap(img, f)
                #     cv2.imwrite(os.path.join(output_path, os.path.basename(img_name)), cam)
                feature_crops = feature_affinity
                patch_mean = feature_crops.mean(-1).unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'channel_mean':
                feature_crops = feature_c5.flatten(2)
                patch_mean = feature_crops.mean(1).unsqueeze(1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_mean).mean(dim=(-1, -2))  # for ID: .mean(dim=-2)
            elif self.mode == 'median':
                feature_crops = feature_c5.flatten(2)
                patch_median = feature_crops.median(-1)[0].unsqueeze(-1)  # (N, C, H*W) -> (N, C)
                patch_sim = torch.abs(feature_crops - patch_median).flatten(1).median(dim=-1)[0]  # for ID: .mean(dim=-2)
            elif self.mode =='std':
                patch_sim = feature_c5.std(dim=(-1,-2)).mean(-1)
            else:
                raise NotImplementedError

        return patch_sim, type
