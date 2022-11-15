import os

method_list = ["GradNormBatch", "MSP", "Energy", "ODIN"]
# method_list = ["GradNormBatch", "MSPCustom", "EnergyCustom", "ODINCustom"]
method_name = method_list[1]
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
    # temperature=1,
    target_file=training_file,
    # target_noise=2,
    classifier=dict(
        type='VitClassifier',
        model='vit_base_patch16_224_in21k',
        checkpoint='/data/csxjiang/vit_lt_ckpt/vit-b16-lt-a8-ep90.pth.tar',
        num_classes=21843,
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
