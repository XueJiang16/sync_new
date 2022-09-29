#gits
#bash ./tools/ood_dist_test.sh custom_config/resnet101.py foo 2 
#python ./tools/ood_test.py custom_config/resnet50.py foo 
#bash ./tools/ood_dist_test.sh custom_config/resnet101_nest_cfg.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_uniform.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_gaussian.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_colorband.py foo 
bash ./tools/ood_dist_test.sh custom_config/resnet50.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/resnet50_official.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/meanstd.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/patchsim.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/inat_ib/resnet101.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/aug_contrast.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/cifar10.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/threshold_act.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/vis_features.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/multi_modal/clip.py foo 2 
#bash ./tools/dist_precompute.sh custom_config/dice_precompute.py /data/csxjiang/ood_ckpt/ckpt/LT_repeat3_a8/epoch_100.pth 2 --metrics accuracy
#bash ./tools/ood_dist_test.sh custom_config/resnet101.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/mobilenet3_large.py foo 2
