from model_config import model as model_cfg
from data_process import prepare_data, prepare_data_zooming, transform_imagenet

import torch
from mmcls.models import build_ood_model
import sys
import os


input = sys.argv[-1]
model = build_ood_model(model_cfg)
model.init_weights()
model.eval()

scores = []
scale_factor = []
with torch.no_grad():
    for i in range(10, 21):
        x = prepare_data_zooming(input, scale_factor=i / 10)
        score, _ = model(**x)
        scores.append(float(score[0]))
        scale_factor.append(i / 10)

import matplotlib.pyplot as plt
plt.plot(scale_factor, scores)
plt.savefig("score_wrt_scale_{}.png".format(os.path.basename(os.path.splitext(input)[0])))