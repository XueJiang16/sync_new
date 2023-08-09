#
#GPU=1
#CPU=1
#node=63
#jobname=openood
#
#PYTHONPATH='.':$PYTHONPATH \
#python main.py \
#    --config configs/datasets/imagenet/imagenet.yml \
#    configs/datasets/imagenet/imagenet_ood.yml \
#    configs/networks/resnet50.yml \
#    configs/pipelines/test/test_ood.yml \
#    configs/preprocessors/base_preprocessor.yml \
#    configs/postprocessors/fsta.yml \
#    --num_workers 4 \
#    --ood_dataset.image_size 256 \
#    --dataset.test.batch_size 256 \
#    --dataset.val.batch_size 256 \
#    --network.pretrained True \
#    --network.checkpoint './results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
#    --merge_option merge

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor fsta \
    --save-score --save-csv #--fsood

# full-spectrum ood
#python scripts/eval_ood_imagenet.py \
#    --tvs-pretrained \
#    --arch resnet50 \
#    --postprocessor ash \
#    --save-score --save-csv --fsood
