method_name = "KNN"
model_name = 'resnet18'
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
    dumped_feature='/data/csxjiang/knn_cache/spurious_resnet_feature_stat.pth',
    k=1000,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/home/csxjiang/jx/sync/Spurious_OOD/checkpoints/waterbird/erm_r_0_9/erm_r_0_9_20230115/ckpt30.pth'),
        backbone=dict(
            type='ResNetActivation',
            depth=18,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
    )
)
ood_pipeline =[dict(type='Collect', keys=['img', 'type'])]
transform = "ImageNet"
transform2 = 'Cifar'

# aug =
aug = None

data = dict(
    samples_per_gpu=256 if method_name is not 'ODIN' else 32,
    workers_per_gpu=4,
    id_data=dict(
        name='waterbird',
        type='CsvDataset',
        path='/data/csxjiang/spurious_ood/waterbird_complete90_forest2water2',
        pipeline=ood_pipeline,
        # len_limit=5000 if quick_test else -1,
        aug=aug,
    ),
    ood_data=[
        dict(
            name='SVHN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/svhn/images',
            pipeline=ood_pipeline,
            transform=transform,
            len_limit=2000 if quick_test else -1,
        ),
        dict(
            name='LSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/LSUN/test',
            pipeline=ood_pipeline,
            transform=transform,
            len_limit=2000 if quick_test else -1,
        ),
        dict(
            name='iSUN',
            type='FolderDataset',
            path='/data/csxjiang/cifar_benchmark/iSUN/iSUN_patches',
            pipeline=ood_pipeline,
            transform=transform,
            len_limit = 2000 if quick_test else -1,
        ),
        dict(
                name='Placebg',
                type='FolderDataset',
                path='/data/csxjiang/spurious_ood/ood_datasets/placesbg/image_collate',
                pipeline=ood_pipeline,
                len_limit=2000 if quick_test else -1,
                aug=aug,
            ),
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
