# Copyright (c) OpenMMLab. All rights reserved
from mmcv.runner import IterBasedRunner
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import is_seq_of


@HOOKS.register_module()
class EpochHook(Hook):
    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        curr_epoch = runner.epoch
        runner.model.module.head.compute_loss.set_drw_epoch(curr_epoch)
