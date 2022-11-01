import os
pick_class = os.environ['PICK_CLASS']
del os
super_class_names = ["dogs", "other-mammals", "birds", "reptiles_fish_amphibians", "inverterbrates", "food_plants_fungi",
                     "devices", "structures_furnishing", "clothes_covering", "implements_containers_misc-objects", "vehicles"]
num_subclass_per_superclass = [130, 88, 59, 60, 61, 63, 172, 90, 92, 117, 68]
full_train_list = ["/data/csxjiang/meta/superclasses/train_{}.txt".format(x) for x in super_class_names]
full_val_list = ["/data/csxjiang/meta/superclasses/val_{}.txt".format(x) for x in super_class_names]
if pick_class == "":
    raise RuntimeError("Must set PICK_CLASS environment value.")
else:
    pick_class = int(pick_class)
    print("***** PICK_CLASS = {} *****".format(str(pick_class)))
    assert pick_class < len(super_class_names)
train_superclass_idx = [x for x in range(len(super_class_names)) if x != pick_class]
train_num_class = sum([num_subclass_per_superclass[x] for x in train_superclass_idx])
train_list = [full_train_list[x] for x in train_superclass_idx]
val_list = [full_val_list[x] for x in train_superclass_idx]
dummy_class_name = [str(x) for x in range(train_num_class)]


method_list = ['MSP', 'ODIN', 'Energy', 'GradNormBatch', 'ThresholdActivation']
method_name = 'FeatureMapSim'
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = 'val_test'
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
noise_engine = None

model = dict(
    type=method_list[3],
    debug_mode=False,
    num_classes=train_num_class,
    # temperature=1,
    target_file=None,
    # target_noise=0.1,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/validation_ckpt/imagenet/resnet50/epoch_100.pth'),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
            # random_block=[1],
            # random_block_k=[0.1],
            # random_block_location=[2],  ## 0:C2 1:C3 2:C4 3:C5
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=train_num_class,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)

# model = dict(
#     type=method_name,
#     num_crop=3,
#     img_size=224,
#     threshold=0.4,
#     order=1,
#     mode='mean',
#     ood_detector=dict(
#         type=method_list[2],
#         debug_mode=False,
#         num_classes=train_num_class,
#         # temperature=1,
#         target_file=None,
#         classifier=dict(
#             type='ImageClassifier',
#             # init_cfg=dict(type='Pretrained', checkpoint='/home/csxjiang/sync/mmclassification/ckpt/inat/epoch_80.pth'),
#             # init_cfg=None,
#             init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
#             # init_cfg=dict(type='Pretrained',
#             #               checkpoint='/home/csxjiang/sync/mmclassification_dev/resnet50_random_block_rand_like_0.033.pth'),
#             backbone=dict(
#                 type='ResNet',
#                 depth=50,
#                 num_stages=4,
#                 out_indices=(3,),
#                 style='pytorch',
#                 random_block=[1],
#                 random_block_k=[0.14],
#                 random_block_location=[2],  # 0:C2 1:C3 2:C4 3:C5
#             ),
#             # neck=dict(type='GlobalAveragePooling'),
#             neck=dict(type='TopKAveragePooling',
#                       k=0.7),
#             head=dict(
#                 type='LinearClsHead',
#                 num_classes=train_num_class,
#                 in_channels=2048,
#                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#                 topk=(1, 5))
# )
#     )
# )
pipline =[dict(type='Collect', keys=['img', 'type'])]
# aug = ['fog']
aug = None

data = dict(
    samples_per_gpu=256 if method_name is not 'ODIN' else 32,
    workers_per_gpu=4,
    id_data=dict(
        name='SubImageNet',
        type='ImageNetSuperclass',
        path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
        data_ann=val_list,
        pipeline=pipline,
        len_limit=5000 if quick_test else 50000,
    ),
    ood_data=[
        dict(
            name=super_class_names[pick_class],
            type='TxtDataset',
            path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
            data_ann="/data/csxjiang/meta/superclasses/train_{}.txt".format(super_class_names[pick_class]),
            pipeline=pipline,
            len_limit=1000 if quick_test else 10000,
            train_label=None,
            aug=aug,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './result_validation/resnet50_wo_{}/'.format(super_class_names[pick_class])
