from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier
from .utils import print_category, print_topk


@OOD.register_module()
class GradNorm(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1, target_file=None, **kwargs):
        super(GradNorm, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1).to("cuda:{}".format(self.local_rank))
        if target_file is not None:
            cls_idx = []
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    cls_idx.append(int(segs[-1]))
            cls_idx = np.array(cls_idx, dtype='int')
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            target = cls_num / np.sum(cls_num)
            # target[109] += 30000
            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank))

    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        self.classifier.zero_grad()
        img = input['img']
        assert img.shape[0] == 1, "GradNorm backward implementation only supports batch = 1."
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        # print("Self rank: {}, output device = {}".format(self.local_rank, outputs.device))
        # assert False
        # outputs, _ = self.classifier.simple_test(softmax=False, **input)
        targets = self.target
        outputs = outputs / self.temperature
        loss = torch.sum(torch.mean(-targets * self.logsoftmax(outputs), dim=-1))

        loss.backward()
        layer_grad = self.classifier.head.fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        return layer_grad_norm, type

@OOD.register_module()
class GradNormBatch(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1, target_file=None, debug_mode=False, target_noise=0, **kwargs):
        super(GradNormBatch, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        classifier['head']['require_features'] = True
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.debug_mode = debug_mode
        if self.debug_mode:
            if os.environ['LOCAL_RANK'] == '0':
                print("*******DEBUG MODE********")
        if target_file is not None:
            cls_idx = []
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    cls_idx.append(int(segs[-1]))
            cls_idx = np.array(cls_idx, dtype='int')
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            target = cls_num / np.sum(cls_num)
            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank)) / self.num_classes
        if target_noise != 0:
            std_target = self.target.std()
            std_noise = target_noise * std_target
            noise = torch.randn_like(self.target) * std_noise
            self.target += noise
            self.target[self.target < 0] = 0
            self.target = self.target / self.target.sum()


    def forward(self, **input):
        with torch.no_grad():
            if "dataset_name" in input:
                dataset_name = input['dataset_name']
                dump_path = "results/gradnorm_dump/{}".format(dataset_name)
                os.makedirs(dump_path, exist_ok=True)
                del input['dataset_name']
            if "type" in input:
                type = input['type']
                del input['type']
            outputs, features = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            targets = self.target
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048
            if self.debug_mode:
                # print_topk(outputs, softmax=True)
                S_dump = S.cpu().tolist()
                for id_score, filename_ in zip(S_dump, input['img_metas']):
                    filename = os.path.splitext(os.path.basename(filename_['filename']))[0] + ".txt"
                    with open(os.path.join(dump_path, filename), mode='w') as f:
                        f.write(str(id_score)+'\n')
        return S, type

@OOD.register_module()
class GradNormBatchScore(BaseModule):
    def __init__(self, classifier, num_classes, temperature=1, target_file=None, debug_mode=False,**kwargs):
        super(GradNormBatchScore, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        classifier['head']['require_features'] = True
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.debug_mode = debug_mode
        if self.debug_mode:
            if os.environ['LOCAL_RANK'] == '0':
                print("*******DEBUG MODE********")
        if target_file is not None:
            cls_idx = []
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    cls_idx.append(int(segs[-1]))
            cls_idx = np.array(cls_idx, dtype='int')
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            # cls_num[109] += 30000
            target = cls_num / np.sum(cls_num)
            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank)) / self.num_classes

    def forward(self, **input):
        with torch.no_grad():
            outputs, features = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            targets = self.target
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048
        return S, out_softmax.clone()

@OOD.register_module()
class GradNormCos(GradNorm):
    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        self.classifier.zero_grad()
        img = input['img']
        assert img.shape[0] == 1, "GradNorm backward implementation only supports batch = 1."
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        # print("Self rank: {}, output device = {}".format(self.local_rank, outputs.device))
        # assert False
        # outputs, _ = self.classifier.simple_test(softmax=False, **input)
        targets = self.target
        outputs = outputs / self.temperature
        out_softmax = torch.nn.functional.softmax(outputs, dim=1)
        sim = -out_softmax * targets
        sim = sim.sum(1) / (torch.norm(out_softmax, dim=1) * torch.norm(targets, dim=1))
        loss = sim.unsqueeze(1)
        loss.backward()
        layer_grad = self.classifier.head.fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        return layer_grad_norm, type

@OOD.register_module()
class KLDiv(GradNorm):
    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            targets = self.target
            outputs = outputs / self.temperature
            kl_score = torch.mean(-targets * self.logsoftmax(outputs), dim=-1)
        return kl_score, type

