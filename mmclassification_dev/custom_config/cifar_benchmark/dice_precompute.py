precompute_name = '/data/csxjiang/dice_cache/cifar100_resnet_p90_'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DiceHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        info=None,
        p=0.9,
        mode='precompute'))
# model = dict(
#         type='ImageClassifier',
#         # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/densenet121_4xb256_in1k_20220426-07450f99.pth'),
#         backbone=dict(
#             type='DenseNet',
#             arch='121'),
#         neck=dict(type='GlobalAveragePooling'),
#         head=dict(type='DiceHead',
#                     num_classes=1000,
#                     in_channels=1024,
#                     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#                     topk=(1, 5),
#                     info=None,
#                     p=0.5,
#                     mode='precompute')
#     )

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=train_pipeline),
    val=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=test_pipeline,
        test_mode=True))


dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/'
