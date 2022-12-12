# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
import time
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.runner.checkpoint import save_checkpoint

from mmcls.apis import single_gpu_test_ood, single_gpu_test_ood_score, single_gpu_test_ssim
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_ood_model
from mmcls.utils import get_root_logger, setup_multi_processes, gather_tensors, evaluate_all
import torch.distributed as dist
from tools.google_helper.google_sheets_reader import GoogleSheets

def parse_args():
    parser = argparse.ArgumentParser(description='ood test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'ipu'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--token',
        type=int,
        default=42,
        help='token for google sheets identification')
    parser.add_argument(
        '--tab_name',
        default='auto',
        help='tabel name to tab data'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def init_eval(cfg, args, is_init=False):
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if not is_init:
            init_dist(args.launcher, **cfg.dist_params)
    return cfg, distributed

def main(args, task_cfg, is_init=False, gs=None, cfg=None):
    if cfg is None:
        cfg = mmcv.Config.fromfile(args.config)

    for k, v in task_cfg.items():
        try:
            rtv = eval(v)
            if type(rtv) in [list, int, float, dict, tuple]:
                pass
            else:
                v = "'{}'".format(v)
        except:
            v = "'{}'".format(v)
        exec("{} = {}".format(k, v))

    if os.environ['LOCAL_RANK'] == '0':
        print("Evaluating {}...".format(cfg.readable_name))
    cfg, distributed = init_eval(cfg, args, is_init=is_init)
    dist.barrier()
    if os.environ['LOCAL_RANK'] == '0':
        print("Get task: {}".format(task_cfg))
        gs.process_lock()

    cfg.gpu_ids = [int(os.environ['LOCAL_RANK'])]

    dataset_id = build_dataset(cfg.data.id_data)
    dataset_ood = [build_dataset(d) for d in cfg.data.ood_data]
    name_ood = [d['name'] for d in cfg.data.ood_data]

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if args.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'id_data', 'ood_data'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader_id = build_dataloader(dataset_id, **test_loader_cfg)
    data_loader_ood = []
    for ood_set in dataset_ood:
        data_loader_ood.append(build_dataloader(ood_set, **test_loader_cfg))

    model = build_ood_model(cfg.model)
    # if not cfg.model.classifier.type == 'VitClassifier':
    model.init_weights()
    # model.classifier.backbone.change_weights()
    # if os.environ['LOCAL_RANK'] == '0':
    #     save_checkpoint(model.ood_detector.classifier, 'resnet50_random_block.pth')
    # assert False
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # model.to("cuda:{}".format(os.environ['LOCAL_RANK']))

    # init distributed env first, since logger depends on the dist info.
    # logger setup
    if os.environ['LOCAL_RANK'] == '0':
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(cfg.work_dir, '{}_{}.log'.format(cfg.readable_name, timestamp))
        os.makedirs(cfg.work_dir, exist_ok=True)
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level,
                                 logger_name='mmcls')
    if os.environ['LOCAL_RANK'] == '0':
        print()
        print("Processing in-distribution data...")
    # outputs_id, type_id = single_gpu_test_ood_score(model, data_loader_id, 'ID')
    outputs_id, type_id = single_gpu_test_ood(model, data_loader_id, 'ID')
    in_scores = gather_tensors(outputs_id)
    in_scores = np.concatenate(in_scores, axis=0)
    # np.save('{}/imagenet_trainset.npy'.format(cfg.work_dir), in_scores)
    type_id = gather_tensors(type_id)
    type_id = np.concatenate(type_id, axis=0)
    if os.environ['LOCAL_RANK'] == '0':
        print("Average ID score:", in_scores.mean())
    # assert False
    # out_scores_list = []
    result_list = []
    avg_auroc = 0
    avg_fpr95 = 0
    for ood_set, ood_name in zip(data_loader_ood, name_ood):
        if os.environ['LOCAL_RANK'] == '0':
            print()
            print("Processing out-of-distribution data ({})...".format(ood_name))
        # outputs_ood, _ = single_gpu_test_ood_score(model, ood_set, ood_name)
        outputs_ood, _ = single_gpu_test_ood(model, ood_set, ood_name)
        out_scores = gather_tensors(outputs_ood)
        out_scores = np.concatenate(out_scores, axis=0)
        # np.save('patchsim_{}.npy'.format(ood_name), out_scores)
        if os.environ['LOCAL_RANK'] == '0':
            print("Average OOD {} score:".format(ood_name), out_scores.mean())
        # out_scores_list.append(out_scores)
        if os.environ['LOCAL_RANK'] == '0':
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(in_scores, out_scores)
            result_list.extend([auroc, aupr_in, aupr_out, fpr95])
            logger.critical('============Overall Results for {}============'.format(ood_name))
            logger.critical('AUROC: {}'.format(auroc))
            logger.critical('AUPR (In): {}'.format(aupr_in))
            logger.critical('AUPR (Out): {}'.format(aupr_out))
            logger.critical('FPR95: {}'.format(fpr95))
            logger.critical('quick data: {},{},{},{}'.format(auroc, aupr_in, aupr_out, fpr95))
            if 3 not in type_id:
                type_list = ['Head', 'Mid', 'Tail']
                ood_id_prop = len(out_scores) / len(in_scores)
                for i in range(len(type_list)):
                    in_scores_ = in_scores[type_id == i]
                    out_scores_ = out_scores[:int(len(in_scores_)*ood_id_prop)]
                    auroc, aupr_in, aupr_out, fpr95 = evaluate_all(in_scores_, out_scores_)
                    logger.critical('============{} Results for {}============'.format(type_list[i], ood_name))
                    logger.critical('AUROC: {}'.format(auroc))
                    logger.critical('AUPR (In): {}'.format(aupr_in))
                    logger.critical('AUPR (Out): {}'.format(aupr_out))
                    logger.critical('FPR95: {}'.format(fpr95))
                    logger.critical('quick data: {},{},{},{}'.format(auroc, aupr_in, aupr_out, fpr95))
    if os.environ['LOCAL_RANK'] == '0':
        for idx in range(len(data_loader_ood)):
            avg_auroc += result_list[4 * idx + 0] / len(data_loader_ood)
            avg_fpr95 += result_list[4 * idx + 3] / len(data_loader_ood)
        result_list.extend([avg_auroc, avg_fpr95])
        logger.critical('all quick data: '+",".join(list(map(str, result_list))))
    return result_list

if __name__ == '__main__':
    import random
    import copy

    args = parse_args()
    SAMPLE_SPREADSHEET_ID = '1znF0Bjncjk6SSrMOWxMSKstkTrusFg4ETkygLQgMyAc'
    # SAMPLE_TAB_NAME = 'auto'
    SAMPLE_TAB_NAME = args.tab_name
    gs_token = args.token
    local_rank = int(os.environ['LOCAL_RANK'])
    gs = GoogleSheets(SAMPLE_SPREADSHEET_ID, node_rank=local_rank)
    is_init = False
    cfg = mmcv.Config.fromfile(args.config)
    while True:
        task_cfg = gs.grab_task(SAMPLE_TAB_NAME, is_master=True if local_rank == 0 else False, token=gs_token)
        if len(task_cfg) == 0:
            break
        res = main(args, task_cfg, is_init=is_init, gs=gs, cfg=copy.deepcopy(cfg))
        is_init = True
        print("Task {} finished with result {}".format(task_cfg, res))
        if local_rank == 0:
            gs.update_result(res)
    print("All tasks have been completed, exit.")
