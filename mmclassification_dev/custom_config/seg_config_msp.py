method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'FeatureReweight']
method_name = method_list[0]
model_name = 'resnet50_seg'
train_dataset = 'Balance'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
training_file = None
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    classifier=dict(
        type='ImageSegmentation',
        segmentor=dict(
            init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/ade20k/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'),
            backbone=dict(
                type='ResNetV1c',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                dilations=(1, 1, 2, 4),
                strides=(1, 2, 1, 1),
                norm_cfg=norm_cfg,
                norm_eval=False,
                style='pytorch',
                contract_dilation=True),
            decode_head=dict(
                type='FCNHead',
                in_channels=2048,
                in_index=3,
                channels=512,
                num_convs=2,
                concat_input=True,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
            ),
    )
)  
# pipline =[dict(type='Collect', keys=['img'])]
pipline =[dict(type='Collect', keys=['img', 'type'])]

# aug =
aug = None
input_size = 512
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    id_data=dict(
        name='ADE20K',
        type='FolderDataset',
        path='/data/csxjiang/ood_data/ADE20K/ADEChallengeData2016/images/validation',
        pipeline=pipline,
        input_size=input_size,

        len_limit=2000 if quick_test else -1,
        aug=aug,
    ),
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/iNaturalist/images',
            pipeline=pipline,
            aug=aug,
            input_size=input_size,

            len_limit=400 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline,
            aug=aug,
            input_size=input_size,

            len_limit=400 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline,
            input_size=input_size,

            aug=aug,
            len_limit=400 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            aug=aug,
            input_size=input_size,

            len_limit=400 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/1031'
