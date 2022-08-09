method_name = 'KLDiv'
model_name = 'resnet50'
train_dataset = 'cifar10'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = False
training_file = None
# training_file = '/data/csxjiang/meta/train_LT_a8.txt'
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=10,
    # temperature=1,
    target_file=training_file,
    # target_noise=0.1,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='~/sync/mmclassification/ckpt/res50_pretrain21k_cifar10/epoch_200.pth'),
        backbone=dict(
            type='ResNet_CIFAR',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))))

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'type'])
]

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='cifar10',
        type='CIFAR10',
        data_prefix='/data/csxjiang/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    ood_data=[dict(
        name='cifar100',
        type='CIFAR100',
        data_prefix='/data/csxjiang/cifar100',
        pipeline=test_pipeline,
        test_mode=True)])


dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/resnet50_cifar10'

