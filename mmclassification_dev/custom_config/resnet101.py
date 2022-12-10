import os

# method_list = ["GradNormBatch", "MSP", "Energy", "ODIN"]
method_list = ["GradNormBatch", "MSPCustom", "EnergyCustom", "ODINCustom"]
method_name = method_list[-1]
model_name = 'resnet101'
train_dataset = 'a8'
# info=os.path.expanduser('/data/csxjiang/dice_cache/imagenet_res101_{}_feature_stat.pth'.format(train_dataset))

custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
# training_file = None
training_file = '/data/csxjiang/meta/train_LT_{}.txt'.format(train_dataset)
model = dict(
    type=method_name,
    num_classes=1000,
    debug_mode = False,
    # temperature=1,
    target_file=training_file,
    # target_noise=2,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/ood_ckpt_other/LT_{}/epoch_100.pth'.format(train_dataset)),
        # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/ood_ckpt_other/resnet101_imagnet10%_100e.pth'),
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
        # head = dict(
        #     type='DiceHead',
        #     num_classes=1000,
        #     in_channels=2048,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5),
        #     info=info,
        #     p=0.7, )
    )
)
# pipline =[dict(type='Collect', keys=['img'])]
pipline =[dict(type='Collect', keys=['img', 'type'])]
data = dict(
    samples_per_gpu=256 if "ODIN" not in method_name else 32,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        # train_label=training_file,
    ),
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
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/imagenet_a8'
