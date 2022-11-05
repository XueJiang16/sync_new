method_name = 'ODIN'
model_name = 'resnet101'
train_dataset = 'Balance'
custom_name = "Official"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    temperature=1,
    # target_file='/data/csxjiang/meta/train_LT_a8.txt',
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet101_8xb32_in1k_20210831-539c63f8.pth'),
        backbone=dict(
            type='ResNet',
            depth=101,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)
pipline =[
    dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        pipeline=pipline),
    # id_data=dict(
    #     type='JsonDataset',
    #     path='/data/csxjiang/',
    #     data_ann='/data/csxjiang/ood_data/inat/val2018.json',
    #     pipeline=[
    #         dict(type='LoadImageFromFile'),
    #         dict(type='Resize', size=480),
    #         dict(
    #             type='Normalize',
    #             mean=[123.675, 116.28, 103.53],
    #             std=[58.395, 57.12, 57.375],
    #             to_rgb=True),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img'])
    #     ]),
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/iNaturalist/images',
            pipeline=pipline),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/'
