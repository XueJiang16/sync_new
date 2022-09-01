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
    def __init__(self, classifier, num_classes, target_file=None, **kwargs):
        super(FeatureVis, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()

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
            out_dir = os.path.join('./vis_cam_features/', mid_path)
            os.makedirs(out_dir, exist_ok=True)
            _, C4_features = self.classifier(return_loss=False, softmax=False, post_process=False,
                                             require_backbone_features_idx='0', **input)
            k = 0.1
            C4_features[C4_features<k] = 0
            C4_features[C4_features>=k] = 1
            C4_features = C4_features.mean(1).cpu().numpy()
            C4_features_std = (C4_features - 0.079) / (0.1953-0.079)
            C4_features_std[C4_features_std>1]=1
            C4_features_std[C4_features_std<0]=0
            # C4_features_std = (C4_features - C4_features.min((1,2))[:,None,None]) / (C4_features.max((1,2))-C4_features.min((1,2)))[:,None,None]
            for i in range(len(filenames)):
                try:
                    img = cv2.imread(filenames[i])
                    res = show_heatmap(img, C4_features_std[i])
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

