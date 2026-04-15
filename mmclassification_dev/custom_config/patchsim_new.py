# 基础参数
method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'ThresholdActivation']
method_name = 'FeatureMapSim'
model_name = 'resnet50'
custom_name = "Official"
train_dataset = 'ImageNet'
num_classes = 1000
num_classes_ = 100  # 仅保留变量格式

# 名称生成
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)

quick_test = False
noise_engine = None

# ===================== 模型 =====================
model = dict(
    type=method_name,
    num_crop=3,
    img_size=224,
    threshold=0.1,
    order=1,
    mode='mean',
    fuse_const=0.1,
    ood_detector=dict(
        type=method_list[0], 
        debug_mode=False,
        num_classes=num_classes,
        target_file=None,
        classifier=dict(
            type='ImageClassifier',
            # ImageNet 标准 ResNet50 预训练权重
            init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/mmcls_offical/resnet50_8xb32_in1k_20210831-ea4938fc.pth'),
            backbone=dict(
                type='ResNetActivation',  # ImageNet 用标准 ResNet，不是 CIFAR 版
                depth=50,
                num_stages=4,
                # TA 参数
                th_act_k=0.4,
                th_act_stage=2,
                th_act_location=5,
                # FMS 参数
                feature_sim_stage=3,
                feature_sim_location=2,
                out_indices=(3,),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=num_classes,
                in_channels=2048,  # ResNet50 输出维度
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5)),
        )
    )
)

# ===================== 数据管道 =====================
ood_pipeline = [
    dict(type='Collect', keys=['img', 'type'])
]
transform = "ImageNet"

# ===================== 数据集 =====================
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    # ID 数据：ImageNet Val
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        # path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
        # data_ann='/data/csxjiang/meta/train_labeled.txt',
        pipeline=ood_pipeline,
        len_limit=5000 if quick_test else -1,
        train_label=None,
        aug=None,
    ),

    # OOD 标准 4 个基准
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/iNaturalist/images',
            pipeline=ood_pipeline,
            transform=transform
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=ood_pipeline,
            transform=transform
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=ood_pipeline,
            transform=transform
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=ood_pipeline,
            transform=transform
        ),
    ],
)

# ===================== 运行配置 =====================
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
work_dir = './results/imagenet_resnet50_fms'