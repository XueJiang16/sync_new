from model_config import model as model_cfg
from data_process import prepare_data, transform_imagenet

import torch
from mmcls.models import build_ood_model


input = "/data/csxjiang/val/ILSVRC2012_val_00024946.JPEG"
model = build_ood_model(model_cfg)
model.init_weights()
model.eval()

with torch.no_grad():
    x = prepare_data(input, transform_imagenet)
    score, _ = model(x)
    print(score)