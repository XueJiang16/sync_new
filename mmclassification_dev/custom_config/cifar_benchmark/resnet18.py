import os

info=os.path.expanduser('/data/csxjiang/dice_cache/cifar100_resnet_p90_feature_stat.pth')
method_list = ["GradNormBatch", "MSP", "Energy", "ODIN"]

method_name = method_list[2]
model_name = 'resnet18'
custom_name = "Official"
train_dataset = 'cifar_100'
num_classes = int(train_dataset.split('_')[-1])
num_classes_ = 10
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=num_classes,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained',
                      # checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'),
                      checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/cifar100_resnet18.pth'),
    backbone=dict(
            type='ResNet_CIFAR',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        # head=dict(
        #     type='LinearClsHead',
        #     num_classes=num_classes,
        #     in_channels=512,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5)),
        # head=dict(
        #     type='DiceHead',
        #     num_classes=num_classes,
        #     in_channels=512,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5),
        #     info=info,
        #     p=0.9,),
        head=dict(
            type='ReactHead',
            num_classes=num_classes,
            in_channels=512,
            threshold=1,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))

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
        name='cifar{}'.format(num_classes_),
        type='CIFAR{}OOD'.format(num_classes_),
        data_prefix='/data/csxjiang/cifar{}'.format(num_classes_),
        transform=transform,
        pipeline=ood_pipeline,
        test_mode=True
        )
        # dict(
        #     name='SVHN',
        #     type='FolderDataset',
        #     path='/data/csxjiang/cifar_benchmark/svhn/images',
        #     pipeline=ood_pipeline,
        #     transform=transform,
        # ),
        # dict(
        #     name='LSUN',
        #     type='FolderDataset',
        #     path='/data/csxjiang/cifar_benchmark/LSUN/test',
        #     pipeline=ood_pipeline,
        #     transform=transform,
        # ),
        # dict(
        #     name='iSUN',
        #     type='FolderDataset',
        #     path='/data/csxjiang/cifar_benchmark/iSUN/iSUN_patches',
        #     pipeline=ood_pipeline,
        #     transform=transform,
        # ),
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
work_dir = './results/cifar'
