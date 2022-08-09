from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter
import json

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import add_noise

@OOD.register_module()
class MSP(BaseModule):
    def __init__(self, classifier, **kwargs):
        super(MSP, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            confs, _ = torch.max(out_softmax, dim=-1)
        return confs, type

@OOD.register_module()
class MSPCustom(BaseModule):
    def __init__(self, classifier, num_classes, target_file=None,target_noise=0,**kwargs):
        super(MSPCustom, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        classifier['head']['require_features'] = True
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        if target_file is not None:
            # cls_idx = []
            # with open(target_file, 'r') as f:
            #     for line in f.readlines():
            #         segs = line.strip().split(' ')
            #         cls_idx.append(int(segs[-1]))
            # cls_idx = np.array(cls_idx, dtype='int')
            # label_stat = Counter(cls_idx)
            # cls_num = [-1 for _ in range(num_classes)]
            # for i in range(num_classes):
            #     cat_num = int(label_stat[i])
            #     cls_num[i] = cat_num
            # cls_num = [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
            # cls_num = [500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286, 273, 260, 248, 237, 226, 216,
            #            206, 197, 188, 179, 171, 163, 156, 149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85,
            #            81, 77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36, 35, 33, 32, 30, 29, 27,
            #            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8,
            #            8, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5]
            # target = cls_num / np.sum(cls_num)
            # target = [0.40870643, 0.24261187, 0.1396526,  0.09073718, 0.05235528, 0.02874834, 0.01950981, 0.01089936, 0.00369978, 0.00307862]
            # target = [0.04755090922117233, 0.045566413551568985, 0.04674213007092476, 0.04288483411073685, 0.04729645326733589, 0.03862952068448067, 0.038489896804094315, 0.03444191813468933, 0.03307860717177391, 0.029994957149028778, 0.03131995350122452, 0.028695467859506607, 0.03006538189947605, 0.02548849582672119, 0.022644972428679466, 0.020814714953303337, 0.023178759962320328, 0.02113150805234909, 0.02100583352148533, 0.01641075685620308, 0.01866854541003704, 0.019027844071388245, 0.014885436743497849, 0.017723876982927322, 0.01435203105211258, 0.014550130814313889, 0.012301989831030369, 0.013718171045184135, 0.011450646445155144, 0.009820462204515934, 0.011735214851796627, 0.011351879686117172, 0.010290280915796757, 0.008601469919085503, 0.008910240605473518, 0.005740542896091938, 0.009976680390536785, 0.00802756566554308, 0.007268883287906647, 0.008052503690123558, 0.005567837040871382, 0.007673847023397684, 0.006782970391213894, 0.005017424002289772, 0.004665859509259462, 0.005841223988682032, 0.0033539568539708853, 0.004800840280950069, 0.00451686093583703, 0.004432307090610266, 0.003226662054657936, 0.00313618965446949, 0.004142226651310921, 0.002465407131239772, 0.003186014946550131, 0.0031393764074891806, 0.0034164292737841606, 0.003323896322399378, 0.0024803567212074995, 0.0035294722765684128, 0.002341219922527671, 0.001110368175432086, 0.002964477753266692, 0.0005593060632236302, 0.0024326404090970755, 0.001509848516434431, 0.0018244509119540453, 0.001494899159297347, 0.0024844978470355272, 0.001275711925700307, 0.0013028749963268638, 0.0006600702763535082, 0.00025188870495185256, 0.0016626775031909347, 0.00013720647257287055, 0.0011001044185832143, 0.0010843542404472828, 0.0005447243456728756, 0.0009894600370898843, 0.0003466836642473936, 0.00014905017451383173, 0.0008332434226758778, 0.0007619873504154384, 0.00031127585680224, 6.598068284802139e-05, 0.0004377042641863227, 0.00040127983083948493, 0.00018349598394706845, 0.0009549130918458104, 0.00013694827794097364, 0.0005274583236314356, 4.248924960847944e-05, 0.0006339326500892639, 0.00010324336471967399, 0.00032026306143961847, 3.155545346089639e-05, 0.00038130211760289967, 1.6452195268357173e-05, 9.576500451657921e-05, 0.00094736332539469]
            target = json.load(open(target_file))


            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank)) / self.num_classes
        if target_noise != 0:
            self.target = add_noise(self.target, target_noise)

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs, features = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            targets = self.target
            confs = out_softmax - targets
            # cos_sim = -out_softmax * targets
            # cos_sim = cos_sim.sum(1) / (torch.norm(out_softmax, dim=1) * torch.norm(targets, dim=1))
            # cos_sim = (1 + cos_sim) / 2
            # cos_sim = cos_sim.unsqueeze(1)
            # confs = confs * cos_sim
            confs, _ = torch.max(confs, dim=-1)
        return confs, type




