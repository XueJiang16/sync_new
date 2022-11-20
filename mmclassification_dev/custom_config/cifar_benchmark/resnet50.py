# import os
#
# info=os.path.expanduser('/data/csxjiang/dice_cache/imagenet_a8_feature_stat.pth')

method_name = 'MSP'
model_name = 'resnet50'
custom_name = "Official"
train_dataset = 'cifar_10'
num_classes = int(train_dataset.split('_')[-1])
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = False
# training_file = '/data/csxjiang/meta/train_labeled.txt'
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=num_classes,
    temperature=1,
    # target_file='/data/csxjiang/meta/train_LT_a8.txt',
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained',
                      checkpoint='~/sync/mmclassification/ckpt/res50_pretrain21k_cifar10/epoch_200.pth'),
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=num_classes,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
        # head=dict(
        #     type='DiceHead',
        #     num_classes=1000,
        #     in_channels=2048,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5),
        #     info=info,
        #     p=0.7,)
    )
)
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'type'])
]

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='cifar{}'.format(num_classes),
        type='CIFAR{}'.format(num_classes),
        data_prefix='/data/csxjiang/cifar{}'.format(num_classes),
        pipeline=pipeline,
        test_mode=True),
    ood_data=[
        dict(
            name='SVHN',
            type='SVHN',
            path='/data/csxjiang/cifar_benchmark/svhn',
            split='test',
            pipeline=pipeline,
            download=False,
        ),
        dict(
            name='LSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/LSUN/test',
            pipeline=pipeline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='iSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/iSUN/iSUN_patches',
            pipeline=pipeline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipeline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipeline,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results'
