method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'ThresholdActivation']
method_name = 'FeatureMapSim'
# method_name = 'FeatureReweight'
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = 'fc_th_act'
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = False
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
            # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet34_8xb32_in1k_20210831-f257d4e6.pth'),
            init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3,),
                style='pytorch',
                random_block=[1],
                random_block_k=[0.15],
                random_block_location=[2],  # 0:C2 1:C3 2:C4 3:C5
            ),
            neck=dict(type='GlobalAveragePooling'),
            # neck=dict(type='TopKAveragePooling',
            #           k=k_c5),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5))
            # head=dict(
            #     type='ReactHead',
            #     threshold=2,
            #     num_classes=1000,
            #     in_channels=2048,
            #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            #     topk=(1, 5))
)
    )
)
