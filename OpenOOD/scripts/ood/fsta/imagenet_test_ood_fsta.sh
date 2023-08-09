
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
