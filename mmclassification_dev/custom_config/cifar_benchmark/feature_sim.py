method_name = 'FeatureReweight'
model_name = 'resnet18'
custom_name = "Official"
train_dataset = 'cifar_100'
num_classes = int(train_dataset.split('_')[-1])
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)

# 18: (BasicBlock, (2, 2, 2, 2)),
#         34: (BasicBlock, (3, 4, 6, 3)),
#         50: (Bottleneck, (3, 4, 6, 3)),
#         101: (Bottleneck, (3, 4, 23, 3)),
#         152: (Bottleneck, (3, 8, 36, 3))

model = dict(
    type=method_name,
    num_classes=num_classes,
    mode='mean',
    ood_detector=dict(
        type='MSP',
        num_classes=num_classes,
        classifier=dict(
            type='ImageClassifier',
            init_cfg=dict(type='Pretrained',
                          # checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'),
                          # checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'),
                          checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/cifar100_resnet18.pth'),
            backbone=dict(
                type='ResNet_CIFAR',
                depth=18,
                num_stages=4,
                th_act_k=0.2,
                th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
                th_act_location=1, ## No. of conv layer
                feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
                feature_sim_location=2,  ## No. of conv layer
                out_indices=(3,),
                style='pytorch'),
            # neck=dict(type='GlobalAveragePooling'),
            # head=dict(
            #     type='LinearClsHead',
            #     num_classes=num_classes,
            #     in_channels=512,
            #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            #     topk=(1, 5))
        )
    )
)
ood_pipeline = [
    dict(type='Collect', keys=['img', 'type'])
]

transform = "Cifar"

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='cifar{}'.format(num_classes),
        type='CIFAR{}OOD'.format(num_classes),
        data_prefix='/data/csxjiang/cifar{}'.format(num_classes),
        transform=transform,
        pipeline=ood_pipeline,
        test_mode=True),
    ood_data=[
        dict(
            name='SVHN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/svhn/images',
            pipeline=ood_pipeline,
            transform=transform,
        ),
        dict(
            name='LSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/LSUN/test',
            pipeline=ood_pipeline,
            transform=transform,
        ),
        dict(
            name='iSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/iSUN/iSUN_patches',
            pipeline=ood_pipeline,
            transform=transform,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=ood_pipeline,
            transform=transform,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=ood_pipeline,
            transform=transform,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/cifar'
