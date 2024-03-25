method_name = 'ScalableClassifier'
model_name = 'ViT-B/16'
#  ['RN50', 224*224
#  'RN101', 224*224
#  'RN50x4', 288*288
#  'RN50x16', 384*384
#  'RN50x64', 448*448
#  'ViT-B/32', 224*224
#  'ViT-B/16', 224*224
#  'ViT-L/14', 224*224
#  'ViT-L/14@336px'] 336*336

train_dataset = 'Balance'
custom_name = "Official"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name).replace('/','_')
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset).replace('/','_')
quick_test = False

model = dict(
    type=method_name,    debug_mode=False,
    # dump_score="/data/csxjiang/score_analysis/neg_label/msp",
    t=1,
    ngroup=1,
    group_fuse_num=None,
    classifier=dict(
        type='CLIPScalableClassifier',
        # type='HuggingFaceCLIPScalableClassifier',
        arch=model_name,
        train_dataset='imagenet',
        wordnet_database='/data/csxjiang/wordnet/txtfiles/',
        # wordnet_database='/data/csxjiang/wordnet/ood_gt_labels/',
        # wordnet_database='/data/csxjiang/result_dump/dummy_wordnet_database/',
        txt_exclude='noun.animal.txt,noun.food.txt',
        #txt_exclude=None,
        neg_subsample=-1,
        pos_neg_sim='neg_centroid',
        neg_topk=0.15, # percentage
        emb_batchsize=1000,
        prompt_idx_pos=85, #[0,80]
        prompt_idx_neg=85,
        dump_neg=False,
        load_dump_neg=False,
        pencentile=0.95,
        # pos_topk=1,
    )
)
pipline =[
          dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        # path='/data/csxjiang/ILSVRC/Data/CLS-LOC/train',
        # data_ann='/data/csxjiang/meta/train_labeled.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        train_label=None,
    ),
    # id_data=dict(
    # name='ImageNet-Sketch',
    # type='FolderDataset',
    # path='/data/csxjiang/domain_shift_data/sketch_collate',
    # pipeline=pipline,
    # len_limit=1000 if quick_test else -1,
    # ),
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
work_dir = './results/'

