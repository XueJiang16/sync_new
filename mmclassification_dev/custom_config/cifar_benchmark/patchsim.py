method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'ThresholdActivation']
method_name = 'FeatureMapSim'
# method_name = 'FeatureReweight'
model_name = 'resnet18'
custom_name = 'fc_th_act'
train_dataset = 'cifar_10'
num_classes = int(train_dataset.split('_')[-1])
num_classes_ = 100
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
noise_engine = None
# k_c5 = 0.65
model = dict(
    type=method_name,
    num_crop=3,
    img_size=224,
    threshold=0.4,
    order=1,
    # k=k_c5,
    mode='mean',
    fuse_const=0.016,
    ood_detector=dict(
        type=method_list[2],
        debug_mode=False,
        num_classes=1000,
        # temperature=1,
        target_file=None,
        classifier=dict(
            type='ImageClassifier',
            init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/cifar/resnet18_b16x8_cifar10_20210528-bd6371c8.pth'),
            # backbone=dict(
            #     type='ResNet',
            #     depth=18,
            #     num_stages=4,
            #     out_indices=(3,),
            #     style='pytorch',
            #     random_block=[1],
            #     random_block_k=[0.15],
            #     random_block_location=[2],  # 0:C2 1:C3 2:C4 3:C5
            # ),
            backbone=dict(
                type='ResNetActivation',
                depth=18,
                num_stages=4,
                th_act_k=0.15,
                th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
                th_act_location=1,  ## No. of conv layer
                feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
                feature_sim_location=0,  ## No. of conv layer
                out_indices=(3,),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling'),
            # neck=dict(type='TopKAveragePooling',
            #           k=k_c5),
            head=dict(
                type='LinearClsHead',
                num_classes=num_classes,
                in_channels=512,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5))
)
    )
)
ood_pipeline =[dict(type='Collect', keys=['img', 'type'])]
# aug = ['fog']
aug = None
transform='Cifar'
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
work_dir = './results/1031'