method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'ThresholdActivation']
method_name = 'FeatureMapSim'
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = 'fc_th_act'
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
noise_engine = None
model = dict(
    type=method_name,
    num_crop=3,
    img_size=224,
    threshold=0.4,
    order=1,
    mode='mean',
    ood_detector=dict(
        type=method_list[3],
        debug_mode=False,
        num_classes=1000,
        # temperature=1,
        target_file=None,
        classifier=dict(
            type='ImageClassifier',
            # init_cfg=dict(type='Pretrained', checkpoint='/home/csxjiang/sync/mmclassification/ckpt/inat/epoch_80.pth'),
            # init_cfg=None,
            init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
            # init_cfg=dict(type='Pretrained',
            #               checkpoint='/home/csxjiang/sync/mmclassification_dev/resnet50_random_block_rand_like_0.033.pth'),
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3,),
                style='pytorch',
                random_block=[1],
                random_block_k=[0.1],
                random_block_location=[2],  # 0:C2 1:C3 2:C4 3:C5
            ),
            # neck=dict(type='GlobalAveragePooling'),
            neck=dict(type='TopKAveragePooling',
                      k=0.5),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5))
)
    )
)
pipline =[dict(type='Collect', keys=['img', 'type'])]
# aug = ['fog']
aug = None

data = dict(
    samples_per_gpu=256 if method_name is not 'ODIN' else 32,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        # path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
        # data_ann='/data/csxjiang/meta/train_labeled.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        train_label=None,
        aug=aug,
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
            aug=aug,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline,
            aug=aug,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline,
            aug=aug,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            aug=aug,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/1031'