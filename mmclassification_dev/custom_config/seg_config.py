method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'FeatureReweight']
method_name = method_list[-1]
model_name = 'resnet50_seg'
train_dataset = 'Balance'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
training_file = None

model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    # temperature=1,
    target_file=training_file,
    mode='mean',
    ood_detector=dict(
        type=method_list[0],
        classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/ade20k/iter_80000.pth'),
        backbone=dict(
            type='ResNetActivation',
            depth=50,
            num_stages=4,
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_eval=False,
            # contract_dilation=True,
            out_indices=(3,),
            style='pytorch',
            th_act_k=0.2,
            th_act_stage=2,  ## 0:C2 1:C3 2:C4 3:C5
            th_act_location=5, ## No. of conv layer
            feature_sim_stage=3,  ## 0:C2 1:C3 2:C4 3:C5
            feature_sim_location=2,  ## No. of conv layer
        ),
        # neck=dict(type='GlobalAveragePooling'),
        # head=dict(
        #     type='LinearClsHead',
        #     num_classes=1000,
        #     in_channels=2048,
        #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #     topk=(1, 5))
    )
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
