method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'FeatureReweight']
method_name = method_list[2]
model_name = 'resnet18'
train_dataset = 'Balance'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
training_file = None

# 18: (BasicBlock, (2, 2, 2, 2)),
#         34: (BasicBlock, (3, 4, 6, 3)),
#         50: (Bottleneck, (3, 4, 6, 3)),
#         101: (Bottleneck, (3, 4, 23, 3)),
#         152: (Bottleneck, (3, 8, 36, 3))

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
#         init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
#         # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet50_8xb32_in1k_20210831-ea4938fc.pth'),
#         # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet34_8xb32_in1k_20210831-f257d4e6.pth'),
#         # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet101_8xb32_in1k_20210831-539c63f8.pth'),
#         backbone=dict(
#             type='ResNetActivation',
#             depth=18,
#             num_stages=4,
#             out_indices=(3,),
#             style='pytorch',
#             th_act_k=0.2,
#             th_act_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
#             th_act_location=0, ## No. of conv layer
#             feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
#             feature_sim_location=1,  ## No. of conv layer
#         ),
#     )
#     )
# )

model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    # temperature=1,
    # target_file='/data/csxjiang/meta/train_LT_a8.txt',
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/home/csxjiang/jx/sync/Spurious_OOD/checkpoints/waterbird/erm_r_0_9/erm_r_0_9_20230115/ckpt30.pth'),
        backbone=dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=2,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)
# pipline =[dict(type='Collect', keys=['img'])]
ood_pipeline =[dict(type='Collect', keys=['img', 'type'])]
transform = "ImageNet"
transform2 = 'Cifar'

# aug =
aug = None

data = dict(
    samples_per_gpu=256 if method_name is not 'ODIN' else 32,
    workers_per_gpu=4,
    id_data=dict(
        name='waterbird',
        type='CsvDataset',
        path='/data/csxjiang/spurious_ood/waterbird_complete90_forest2water2',
        pipeline=ood_pipeline,
        # len_limit=5000 if quick_test else -1,
        aug=aug,
    ),
    ood_data=[
        dict(
            name='SVHN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/svhn/images',
            pipeline=ood_pipeline,
            transform=transform,
            len_limit=2000 if quick_test else -1,
        ),
        dict(
            name='LSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/LSUN/test',
            pipeline=ood_pipeline,
            transform=transform2,
            len_limit=2000 if quick_test else -1,
        ),
        dict(
            name='iSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/iSUN/iSUN_patches',
            pipeline=ood_pipeline,
            transform=transform2,
            len_limit = 2000 if quick_test else -1,
        ),
        dict(
                name='Placebg',
                type='FolderDataset',
                path='/data/csxjiang/spurious_ood/ood_datasets/placesbg/image_collate',
                pipeline=ood_pipeline,
                len_limit=2000 if quick_test else -1,
                aug=aug,
            ),
        # dict(
        #     name='Places',
        #     type='FolderDataset',
        #     path='/data/csxjiang/ood_data/Places/images',
        #     pipeline=ood_pipeline,
        #     transform=transform,
        # ),
        # dict(
        #     name='Textures',
        #     type='FolderDataset',
        #     path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
        #     pipeline=ood_pipeline,
        #     transform=transform,
        # ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/1031'
