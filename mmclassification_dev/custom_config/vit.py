import os

method_list = ["GradNorm", "MSP", "Energy", "ODIN", 'FeatureReweight']
method_name = method_list[-1]
model_name = 'vit'
# info=os.path.expanduser('/data/csxjiang/dice_cache/imagenet_res101_{}_feature_stat.pth'.format(train_dataset))

custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}'.format(method_name, model_name, custom_name)
else:
    readable_name ='{}_{}'.format(method_name, model_name)
quick_test = False
# training_file = None

# for far score
model = dict(
    type=method_name,
    num_classes=1000,
    mode='vit',
    ood_detector=dict(
    type=method_list[1],
    classifier= dict(
    type='ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'),
    backbone=dict(
    type='VisionTransformer',
    arch='b',
    img_size=384,
    patch_size=16,
    output_cls_token=False,
    # out_indices=-1,
    out_indices=10,
    # drop_rate=0.1,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),)
    ))
)

# for baseline
# model = dict(
#     type=method_name,
#     num_classes=1000,
#     classifier= dict(
#     type='ImageClassifier',
#     init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'),
#     backbone=dict(
#     type='VisionTransformer',
#     arch='b',
#     img_size=384,
#     patch_size=16,
#     # drop_rate=0.1,
#     ),
#     neck=None,
#     head=dict(
#         type='VisionTransformerClsHead',
#         num_classes=1000,
#         in_channels=768,
#         loss=dict(
#             type='LabelSmoothLoss', label_smooth_val=0.1,
#             mode='classy_vision'),
#     ))
#)


# pipline =[dict(type='Collect', keys=['img'])]
pipline =[dict(type='Collect', keys=['img', 'type'])]
data = dict(
    samples_per_gpu=256 if "ODIN" not in method_name else 16,
    # samples_per_gpu=1,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        # train_label=training_file,
    ),
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/iNaturalist/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/vit'
