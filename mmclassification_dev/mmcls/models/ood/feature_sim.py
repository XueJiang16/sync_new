from mmcv.runner import BaseModule    # noqa
import torch  # noqa
import torch.nn as nn

import os
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
                feature_crops = feature_c5.flatten(2)
                batch_size, channel, _ = feature_crops.shape
                for i in range(batch_size):
                    for j in range(channel):
                        x = feature_crops[i,j].cpu().detach().numpy()
                        x = x.reshape(-1,1)
                        gmm = GMM(n_components=2, max_iter=1000, random_state=10, covariance_type='full')
                        # find useful parameters
                        mean = gmm.fit(x).means_
                        covs = gmm.fit(x).covariances_
                        weights = gmm.fit(x).weights_

                        # create necessary things to plot
                        x_axis = np.arange(-20, 30, 0.1)
                        y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0]))) * weights[
                            0]  # 1st gaussian
                        y_axis1 = norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0]))) * weights[
                            1]  # 2nd gaussian
                        plt.hist(x, density=True, color='black', bins=np.arange(-100, 100, 1))
                        plt.plot(x_axis, y_axis0, lw=3, c='C0')
                        plt.plot(x_axis, y_axis1, lw=3, c='C1')
                        plt.plot(x_axis, y_axis0 + y_axis1, lw=3, c='C2', ls='dashed')
                        plt.xlim(-10, 20)
                        # plt.ylim(0.0, 2.0)
                        plt.xlabel(r"X", fontsize=20)
                        plt.ylabel(r"Density", fontsize=20)
                        plt.savefig("test.png")
                        plt.close('all')
                        # print(x.shape)
                        assert False



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
