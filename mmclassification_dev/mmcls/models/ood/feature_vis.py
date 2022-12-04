from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import shutil
import cv2

from ..builder import OOD
from mmcls.models import build_classifier

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
class FeatureVis(BaseModule):
    def __init__(self, classifier, classifier_act, num_classes, target_file=None, **kwargs):
        super(FeatureVis, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.classifier_act = build_classifier(classifier_act)
        self.classifier_act.eval()

    def norm(self, features):
        features_mean = features.mean(1)
        features_norm = features_mean / 0.08
        features_norm[features_norm > 1] = 1
        features_norm[features_norm < 0] = 0
        features_norm = features_norm.cpu().numpy()
        return features_norm

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            filenames = [x['filename'] for x in input['img_metas']]
            if 'ood_data' in filenames[0]:
                mid_path = 'OOD'
            else:
                mid_path = 'ID'
            out_dir = os.path.join('./vis_cam_features_new/', mid_path)
            os.makedirs(out_dir, exist_ok=True)
            outputs, C4_features = self.classifier(return_loss=False, softmax=False, post_process=False,
                                             require_backbone_features_idx='0', **input)
            outputs_act, _ = self.classifier_act(return_loss=False, softmax=False, post_process=False,
                                             require_backbone_features_idx='0', **input)
            # energy_confs = torch.logsumexp(outputs, dim=1)
            msp_confs, _ = torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=-1)
            msp_confs_act, _ = torch.max(torch.nn.functional.softmax(outputs_act, dim=1), dim=-1)

            k = 0.1
            # C4_features[C4_features<k] = 0
            # C4_features[C4_features>=k] = 1
            C4_features_norm = self.norm(C4_features)
            C4_features_larger = C4_features.clone()
            C4_features_larger[C4_features_larger < k] = 0
            C4_features_larger = self.norm(C4_features_larger)
            C4_features_lower = C4_features.clone()
            # C4_features_lower[C4_features_lower > k] = 0.08
            # C4_features_lower = self.norm(C4_features_lower)
            C4_features_lower = torch.clamp(C4_features_lower-k, min=0)
            C4_features_lower = self.norm(C4_features_lower)

            # C4_features_larger = C4_features_larger - k


            # print("[{}] mean={}, std={}".format(mid_path, mean, std))
            # C4_features_std = (C4_features - 0.079) / (0.1953-0.079)
            # C4_features_std[C4_features_std>1]=1
            # C4_features_std[C4_features_std<0]=0

            # C4_features_std = (C4_features - C4_features.min((1,2))[:,None,None]) / (C4_features.max((1,2))-C4_features.min((1,2)))[:,None,None]
            for i in range(len(filenames)):
                try:
                    img = cv2.imread(filenames[i])
                    img_cam = img.copy()
                    img_larger = img.copy()
                    img_lower = img.copy()
                    res_cam = show_heatmap(img_cam, C4_features_norm[i])
                    res_larger = show_heatmap(img_larger, C4_features_larger[i])
                    res_lower = show_heatmap(img_lower, C4_features_lower[i])
                    # font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # org = (50, 50)
                    fontScale = 0.8
                    color = (255, 255, 255)
                    thickness = 1
                    img = cv2.putText(img, 'Conf={}'.format(msp_confs[i]), (50, 50), font,
                                        fontScale, color, thickness, cv2.LINE_AA)
                    img = cv2.putText(img, 'Conf={}'.format(msp_confs_act[i]), (50, 80), font,
                                      fontScale, color, thickness, cv2.LINE_AA)
                    res12 = np.hstack([img, res_cam])
                    res34 = np.hstack([res_larger, res_lower])
                    res = np.vstack([res12, res34])
                    # plt.matshow(C4_features[i])
                    filename = os.path.splitext(os.path.basename(filenames[i]))[0]
                    cv2.imwrite(os.path.join(out_dir, '{}_heatmap.jpg'.format(filename)), res)

                except:
                    print('Image Read Error!')
                    continue
                # plt.savefig(os.path.join(out_dir, '{}_heatmap.jpg'.format(filename)))
                # plt.close()
                # shutil.copy(filenames[i], out_dir)
            confs = torch.tensor([0]*len(filenames)).to("cuda:{}".format(self.local_rank))
        return confs, type

