method_name = 'KLDiv'
model_name = 'resnet101'
pipline = [dict(type='Collect', keys=['img', 'type'])]

train_dataset_list = ['LT_a{}'.format(x) for x in range(2, 9)]
target_file_list = ['/data/csxjiang/meta/train_LT_a{}.txt'.format(x) for x in range(2, 9)]
checkpoint_list = ['/data/csxjiang/ood_ckpt/ood_ckpt_other/LT_a{}/epoch_100.pth'.format(x)
                   for x in range(2, 9)]
multi_cfg_list = [train_dataset_list, target_file_list, checkpoint_list]
len_var_cfg = None
for item in multi_cfg_list:
    if len_var_cfg is None:
        len_var_cfg = len(item)
    else:
        assert len(item) == len_var_cfg, "Expect same length of multiple config vars."

multi_cfg = []

for train_data, target_file, ckpt in zip(*multi_cfg_list):
    train_dataset = train_data
    temp_cfg = dict(
        method_name = method_name,
        model_name = model_name,
        train_dataset = train_dataset,
        readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset),
        model = dict(
            type=method_name,
            num_classes=1000,
            temperature=1,
            target_file=target_file,
            classifier=dict(
                type='ImageClassifier',
                init_cfg=dict(type='Pretrained', checkpoint=ckpt),
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
        ),
        data = dict(
            samples_per_gpu=1,
            workers_per_gpu=1,
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

        ),
        dist_params = dict(backend='nccl'),
        log_level = 'CRITICAL',
        # log_level = 'INFO'
        work_dir = './results/rebuttal'
    )
    multi_cfg.append(temp_cfg)