from utils import log
import resnetv2
import torch
import torchvision as tv
import time

import numpy as np
import random
from utils.test_utils import arg_parser, get_measures
import os
import glob

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
import resnet
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from funcs import *
import cv2

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint
import json

class IDDataset(torch.utils.data.Dataset):
    #ImageNet
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)
        # return 2

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label
        # return sample, label, os.path.basename(path)

class IDDataset2(torch.utils.data.Dataset):
    #INaturalist
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append([images_dict[item['image_id']], item['category_id']])
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))

    def __len__(self):
        return len(self.file_list)
        # return 32

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480
    # crop = 224
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                [58.395/255, 57.12/255, 57.375/255]),
    ])
    id_ann = './meta/val_labeled.txt'
    # id_ann = './dataset/ood_data/inat/val2018.json'
    # in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    in_set = IDDataset(args.in_datadir, id_ann, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def run_eval_custom(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()
    try:
        a = float(args.sample_a)
    except TypeError as e:
        print('args.sample_a cannot be converted to float, use 0 as default')
        print(e)
        a = 0
    if a != 0:
        res = sample_function(a)
        cls_num = generate_dis_func(res)
    else:
        cls_idx = []
        label_filename = args.id_cls
        with open(label_filename, 'r') as f:
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

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_msp(in_loader, model)
        # in_scores, id_labels = iterate_data_msp_custom(in_loader, model, target)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_msp(out_loader, model)
        # out_scores, _ = iterate_data_msp_custom(out_loader, model, target)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        # in_scores, id_labels = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin)
        in_scores, id_labels = iterate_data_odin_custom(in_loader, model, args.epsilon_odin, args.temperature_odin, target, mode='linear')
        logger.info("Processing out-of-distribution data...")
        # out_scores, _ = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin)
        out_scores, _ = iterate_data_odin_custom(out_loader, model, args.epsilon_odin, args.temperature_odin, target, mode='linear')
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_energy_custom(in_loader, model, args.temperature_energy, target,
                                                           mode='linear')
        # in_scores, id_labels = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_energy_custom(out_loader, model, args.temperature_energy, target,
                                                   mode='linear')
        # out_scores, _ = iterate_data_energy(out_loader, model, args.temperature_energy)

    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                                   [0, 0, 1, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                                        num_output, magnitude, regressor)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                                 num_output, magnitude, regressor)
    elif args.score == 'GradNorm':
        # in_scores = []
        # out_scores = []
        # with open('LT_a8_in_score.txt', 'r') as f:
        #     for line in f.readlines():
        #         line = line.strip('[')
        #         line = line.strip(']\n')
        #         in_scores.append(line)
        # in_scores = np.array(in_scores, dtype='float32')
        # with open('LT_a8_out_score.txt', 'r') as f:
        #     for line in f.readlines():
        #         line = line.strip('[')
        #         line = line.strip(']\n')
        #         out_scores.append(line)
        # out_scores = np.array(out_scores, dtype='float32')
        logger.info("Processing in-distribution data...")
        in_scores, id_labels = iterate_data_gradnorm_o(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores, _ = iterate_data_gradnorm_o(out_loader, model, args.temperature_gradnorm, num_classes)

    elif args.score == 'new':
        # in_confs, labels = iterate_data_confidence(in_loader, model, isID=True)
        #         # np.savez('id_confidence_score.npz', in_confs=in_confs, labels=labels)
        #         # out_confs = iterate_data_confidence(out_loader, model, isID=False)
        #         # np.savez('ood_confidence_score.npz',out_confs=out_confs)
        #         # exit()
        # cls_idx = []
        # label_filename =args.id_cls

        # logger.info("Processing in-distribution data...")
        # in_scores, id_labels = iterate_data_newplus(in_loader, model, args.temperature_gradnorm, num_classes,target)
        # logger.info("Processing out-of-distribution data...")
        # out_scores, _ = iterate_data_newplus(out_loader, model, args.temperature_gradnorm, num_classes, target)
        in_scores, id_labels = iterate_data_cosine(in_loader, model, target)
        out_scores, _ = iterate_data_cosine(out_loader, model, target)
        # logger.info("Processing in-distribution data...")
        # in_scores, id_labels = iterate_data_cosnorm(in_loader, model, target)
        # logger.info("Processing out-of-distribution data...")
        # out_scores, _ = iterate_data_cosnorm(out_loader, model, target)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    gather_in_scores = [torch.zeros_like(in_scores) - 1 for _ in range(torch.distributed.get_world_size())]
    gather_out_scores = [torch.zeros_like(out_scores)-1 for _ in range(torch.distributed.get_world_size())]
    gather_labels = [torch.zeros_like(id_labels)-1 for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gather_in_scores, in_scores)
    torch.distributed.all_gather(gather_out_scores, out_scores)
    torch.distributed.all_gather(gather_labels, id_labels)
    in_scores = torch.stack(gather_in_scores)
    out_scores = torch.stack(gather_out_scores)
    labels = torch.stack(gather_labels)
    in_scores = in_scores[in_scores != -1]
    out_scores = out_scores[out_scores!=-1]
    labels = labels[labels!=-1]
    in_scores = in_scores.cpu().numpy()
    out_scores = out_scores.cpu().numpy()
    labels = labels.cpu().numpy()
    if args.local_rank == 0:
        # print(labels.shape)
        id_cls = []
        cls_idx = []
        label_filename = args.id_cls
        original_a = label_filename[-5]
        with open(label_filename, 'r') as f:
            for line in f.readlines():
                segs = line.strip().split(' ')
                cls_idx.append(int(segs[-1]))
        cls_idx = np.array(cls_idx, dtype='int')
        res = Counter(cls_idx)
        for item in labels:
            id_cls.append(res[item])
        id_cls = np.array(id_cls)
        in_examples = in_scores.reshape((-1, 1))
        out_examples = out_scores.reshape((-1, 1))
        id_cls = id_cls.reshape((-1, 1))
        # nums = np.random.randint(0, 50000, 10000)
        # in_examples_ = in_examples[nums]
        # id_cls_ = id_cls[nums]
        # plt.figure(1)
        # plt.scatter(in_examples_, id_cls_)
        # plt.savefig('./test_{}_{}.jpg'.format(original_a,a))
        # fake = in_examples>0
        # res = Counter(id_cls[fake].astype('int'))
        # print(res)
        # exit()
        #overall eval
        auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

        #head-tail eval
        head_examples = []
        mid_examples = []
        tail_examples = []
        head_cls = []
        mid_cls = []
        tail_cls = []
        for i in range(in_examples.shape[0]):
            if id_cls[i] >= 100:
                head_examples.append(in_examples[i])
                head_cls.append(id_cls[i])
            elif 20 < id_cls[i] < 100:
                mid_examples.append(in_examples[i])
                mid_cls.append(id_cls[i])
            elif 0 <= id_cls[i] <= 20:
                tail_examples.append(in_examples[i])
                tail_cls.append(id_cls[i])
            else:
                raise ValueError("Unknown id class type {}".format(id_cls[i]))
        out_examples = out_examples.tolist()
        random.shuffle(out_examples)
        auroc_head, aupr_in_head, aupr_out_head, fpr95_head = [0,0,0,0]
        if len(head_examples) > 0:
            head_examples = np.stack(head_examples,axis=0)
            head_ratio = int(head_examples.shape[0] / in_examples.shape[0] * len(out_examples))
            head_out_examples = np.array(out_examples[0:head_ratio])
            # plt.figure(2)
            # plt.scatter(head_examples, head_cls)
            # plt.savefig('./test_head_{}.jpg'.format(a))
            auroc_head, aupr_in_head, aupr_out_head, fpr95_head = get_measures(head_examples, head_out_examples)
            # head_examples_ = head_examples>thr
            # head_acc = np.sum(head_examples_)/len(head_examples_)

        auroc_mid, aupr_in_mid, aupr_out_mid, fpr95_mid = [0,0,0,0]
        if len(mid_examples) > 0:
            mid_examples = np.stack(mid_examples,axis=0)
            mid_ratio = int(mid_examples.shape[0] / in_examples.shape[0] * len(out_examples))
            mid_out_examples = np.array(out_examples[0:mid_ratio])
            # plt.figure(3)
            # plt.scatter(mid_examples, mid_cls)
            # plt.savefig('./test_mid_{}.jpg'.format(a))
            auroc_mid, aupr_in_mid, aupr_out_mid, fpr95_mid = get_measures(mid_examples, mid_out_examples)
            # mid_examples_ = mid_examples > thr
            # mid_acc = np.sum(mid_examples_) / len(mid_examples_)

        auroc_tail, aupr_in_tail, aupr_out_tail, fpr95_tail=[0,0,0,0]
        if len(tail_examples) > 0:
            tail_examples = np.stack(tail_examples,axis=0)
            tail_ratio = int(tail_examples.shape[0] / in_examples.shape[0] * len(out_examples))
            tail_out_examples = np.array(out_examples[0:tail_ratio])
            # plt.figure(4)
            # plt.scatter(tail_examples, tail_cls)
            # plt.savefig('./test_tail_{}.jpg'.format(a))
            auroc_tail, aupr_in_tail, aupr_out_tail, fpr95_tail = get_measures(tail_examples, tail_out_examples)
            # tail_examples_ = tail_examples > thr
            # tail_acc = np.sum(tail_examples_) / len(tail_examples_)
        # in_examples_ = in_examples > thr
        # overall_acc = np.sum(in_examples_) / len(in_examples_)
        # correct_sample = id_cls[in_examples_].astype('int')
        # print(error_sample)

        # res = Counter(correct_sample)
        # info=dict()
        # for k,v in res.items():
        #     info[k] = v
        # with open('res_LT_a4_counter.txt', 'w') as f:
        #     for k, v in info.items():
        #         for img in v:
        #             f.write('{} {}\n'.format(img, k))
        # in_examples_ = in_examples < thr
        # error_sample = id_cls[in_examples_].astype('int')
        # print(error_sample)

        # exit()


        logger.info('============Overall Results for {}============'.format(args.score))
        logger.info('AUROC: {}'.format(auroc))
        logger.info('AUPR (In): {}'.format(aupr_in))
        logger.info('AUPR (Out): {}'.format(aupr_out))
        logger.info('FPR95: {}'.format(fpr95))
        logger.info('quick data: {},{},{},{}'.format(auroc,aupr_in,aupr_out,fpr95))
        # logger.info('============Head Results for {}============'.format(args.score))
        # logger.info('AUROC: {}'.format(auroc_head))
        # logger.info('AUPR (In): {}'.format(aupr_in_head))
        # logger.info('AUPR (Out): {}'.format(aupr_out_head))
        # logger.info('FPR95: {}'.format(fpr95_head))
        # logger.info('quick data: {},{},{},{}'.format(auroc_head,aupr_in_head,aupr_out_head,fpr95_head))
        # logger.info('============Mid Results for {}============'.format(args.score))
        # logger.info('AUROC: {}'.format(auroc_mid))
        # logger.info('AUPR (In): {}'.format(aupr_in_mid))
        # logger.info('AUPR (Out): {}'.format(aupr_out_mid))
        # logger.info('FPR95: {}'.format(fpr95_mid))
        # logger.info('quick data: {},{},{},{}'.format(auroc_mid,aupr_in_mid,aupr_out_mid,fpr95_mid))
        # logger.info('============Tail Results for {}============'.format(args.score))
        # logger.info('AUROC: {}'.format(auroc_tail))
        # logger.info('AUPR (In): {}'.format(aupr_in_tail))
        # logger.info('AUPR (Out): {}'.format(aupr_out_tail))
        # logger.info('FPR95: {}'.format(fpr95_tail))
        # logger.info('quick data: {},{},{},{}'.format(auroc_tail,aupr_in_tail,aupr_out_tail,fpr95_tail))
        logger.flush()


def main(args):
    logger = log.setup_logger(args)
    torch.backends.cudnn.benchmark = True
    if args.score == 'new' or args.score == 'GradNorm':
        args.batch = 1
    # if args.score == 'ODIN':
    #     args.batch = 16
    if args.score == 'Mahalanobis':
        args.batch = 8
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.cuda.set_device(args.local_rank)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    torch.distributed.init_process_group(backend="nccl")

    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    in_sampler = torch.utils.data.distributed.DistributedSampler(in_set)
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size = args.batch, shuffle = False,
        num_workers = args.workers, pin_memory = True, drop_last = False, sampler=in_sampler)
    out_sampler = torch.utils.data.distributed.DistributedSampler(out_set)
    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size = args.batch, shuffle = False,
        num_workers = args.workers, pin_memory = True, drop_last = False, sampler=out_sampler)


    logger.info(f"Loading model from {args.model_path}")

    # model = resnetv2.KNOWN_MODELS[args.model](head_size=1000)
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict_custom(state_dict['model'])
    if 'resnet152' in args.model_path:
        model = resnet.resnet152()
    elif 'resnet50' in args.model_path:
        model = resnet.resnet50()
    elif 'mobile' in args.model_path:
        config = dict(
            type='ImageClassifier',
            backbone=dict(type='MobileNetV3OOD', arch='large'),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='OODHead',
                num_classes=1000,
                in_channels=960,
                mid_channels=[1280],
                dropout_rate=0.2,
                act_cfg=dict(type='HSwish'),
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5)))
        model = build_classifier(config)
        load_checkpoint(model, args.model_path)
    else:
        model = resnet.resnet101()
    if 'mobile' not in args.model_path:
        b = dict()
        a = torch.load(args.model_path)
        for k, v in a["state_dict"].items():
            b[".".join(k.split(".")[1:])] = v
        model.load_state_dict(b)

    # if args.score != 'GradNorm' and args.score != 'new':
    #     model = torch.nn.DataParallel(model)


    model = model.eval().cuda()

    start_time = time.time()
    # run_eval(model, in_loader, out_loader, logger, args, num_classes=1000)
    run_eval_custom(model, in_loader, out_loader, logger, args, num_classes=1000)

    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm', 'new'], default='GradNorm')
    parser.add_argument("--id_cls", default='', help="id label file")
    parser.add_argument("--sample_a", default='0', type=str, help='distribution parameter')

    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    main(parser.parse_args())
