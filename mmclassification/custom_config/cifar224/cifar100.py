model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=[
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=[
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),

    test=dict(
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=[
            dict(type='Resize', size=(224, 224)),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=50)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from=None
resume_from=None
workflow = [('train', 1)]