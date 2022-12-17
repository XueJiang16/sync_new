method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'FeatureReweight']
method_name = method_list[2]
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = False
training_file = None



# model = dict(
#     type=method_name,
#     debug_mode=False,
#     num_classes=1000,
#     # temperature=1,
#     target_file=training_file,
#     mode='mean',
#     ood_detector=dict(
#         type=method_list[0],
#         classifier=dict(
#         type='ImageClassifier',
#         init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/densenet121_4xb256_in1k_20220426-07450f99.pth'),
#         backbone=dict(
#             type='DenseNet',
#             arch='121',
#             th_act_k=0.2,
#             th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
#             th_act_location=23, ## No. of conv layer   # 121: [6, 12, 24, 16]
#             feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
#             feature_sim_location=15,  ## No. of conv layer
#         ),
#     )
#     )
# )

model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    temperature=1,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/densenet121_4xb256_in1k_20220426-07450f99.pth'),
        backbone=dict(
            type='DenseNet',
            arch='121'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1024,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)

# pipline =[dict(type='Collect', keys=['img'])]
pipline =[dict(type='Collect', keys=['img', 'type'])]

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
