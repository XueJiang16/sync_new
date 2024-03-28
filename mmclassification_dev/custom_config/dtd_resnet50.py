# import os
#
# info=os.path.expanduser('/data/csxjiang/dice_cache/imagenet_res50_feature_stat.pth')
method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'FeatureReweight']

method_name = method_list[-1]
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = "Official"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = False
# training_file = '/data/csxjiang/meta/train_labeled.txt'
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=64,
    # temperature=1,
    # target_file='/data/csxjiang/meta/train_LT_a8.txt',
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/textures_ckpt/epoch_36.pth'),
        backbone=dict(
            # type='ResNet',
            # depth=50,
            # num_stages=4,
            # out_indices=(3,),
            # style='pytorch'),
            type='ResNetActivation',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
            th_act_k=0.1,
            th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
            th_act_location=5, ## No. of conv layer
            feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
            feature_sim_location=2,  ## No. of conv layer
        ),
        # neck=dict(type='GlobalAveragePooling'),
        # head=dict(
        #     type='LinearClsHead',
        #     num_classes=64,
        #     in_channels=2048,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5))
        # head=dict(
        #     type='DiceHead',
        #     num_classes=1000,
        #     in_channels=2048,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5),
        #     info=info,
        #     p=0.7,)
        # head=dict(
        #     type='ReactHead',
        #     num_classes=1000,
        #     in_channels=2048,
        #     threshold=1,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5))
    )
)
pipline =[
          dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    id_data=dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
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
                name='ImageNet',
                type='TxtDataset',
                path='/data/csxjiang/val',
                data_ann='/data/csxjiang/meta/val_labeled.txt',
                # path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
                # data_ann='/data/csxjiang/meta/train_labeled.txt',
                pipeline=pipline,
                len_limit=5000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results'
