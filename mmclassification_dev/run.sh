#gits
#bash ./tools/ood_dist_test.sh custom_config/resnet101.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/resnet101_official.py foo 2
#python ./tools/ood_test.py custom_config/resnet50.py foo 
#bash ./tools/ood_dist_test.sh custom_config/resnet101_nest_cfg.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_uniform.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_gaussian.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/Noise/resnet50_colorband.py foo 
#bash ./tools/ood_dist_test.sh custom_config/resnet50.py foo 2
# bash ./tools/ood_dist_test.sh custom_config/resnet50_official.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/meanstd.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/patchsim.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/inat_ib/resnet101.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/aug_contrast.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/cifar10.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/threshold_act.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/vis_features.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/multi_modal/clip.py foo 2 
#bash ./tools/dist_precompute.sh custom_config/dice_precompute.py /data/csxjiang/ood_ckpt/mmcls_offical/densenet121_4xb256_in1k_20220426-07450f99.pth 2 --metrics accuracy
# bash ./tools/dist_precompute.sh custom_config/dice_precompute.py /data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth 2 --metrics accuracy
#bash ./tools/ood_dist_test.sh custom_config/resnet101.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/mobilenet3_large.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/vit_lt.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/patchsim_mobile.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/densenet.py foo 2
# bash ./tools/ood_dist_test.sh custom_config/vit.py foo 2


#bash ./tools/ood_dist_test.sh custom_config/icml_fig/resnet50.py foo 2

## Cifar
#bash ./tools/ood_dist_test.sh custom_config/cifar_benchmark/resnet50.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/cifar_benchmark/resnet18.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/cifar_benchmark/feature_sim.py foo 2



##coco
#bash ./tools/ood_dist_test.sh custom_config/faster_rcnn.py foo 2
# bash ./tools/ood_dist_test_auto.sh custom_config/faster_rcnn.py foo 2 --tab_name coco
#bash ./tools/ood_dist_test_auto.sh custom_config/seg_config.py foo 2 --tab_name ade20k

## google doc
#bash ./tools/ood_dist_test_auto.sh custom_config/patchsim.py foo 2 --tab_name add
#bash ./tools/ood_dist_test_auto.sh custom_config/patch_sim_scale.py foo 2 --tab_name patch_sim_scale
#bash ./tools/ood_dist_test_auto.sh custom_config/patchsim.py foo 2 --tab_name iclr_rebuttal_ssb
#bash ./tools/ood_dist_test_auto.sh custom_config/patchsim_mobile.py foo 2 --tab_name mobile
bash ./tools/ood_dist_test_auto.sh custom_config/resnet50.py foo 2 --tab_name far_fa
#bash ./tools/ood_dist_test_auto.sh custom_config/threshold_act.py foo 2 --tab_name auto2
#bash ./tools/ood_dist_test_auto.sh custom_config/densenet.py foo 2 --tab_name densenet
#bash ./tools/ood_dist_test_auto.sh custom_config/cifar_benchmark/feature_sim.py foo 2 --tab_name cifar18_for_rebuttal
#bash ./tools/ood_dist_test_auto.sh custom_config/cifar_benchmark/resnet18.py foo 2 --tab_name cifar18_for_rebuttal

## spurious
#bash ./tools/dist_precompute.sh custom_config/spurious/dice_precompute.py /home/csxjiang/jx/sync/Spurious_OOD/checkpoints/waterbird/erm_r_0_9/erm_r_0_9_20230115/ckpt30.pth 2 --metrics accuracy
#bash ./tools/dist_precompute.sh custom_config/spurious/knn_precompute.py /home/csxjiang/jx/sync/Spurious_OOD/checkpoints/waterbird/erm_r_0_9/erm_r_0_9_20230115/ckpt30.pth 2 --metrics accuracy
#bash ./tools/ood_dist_test.sh custom_config/spurious/resnet18.py foo 2
#bash ./tools/ood_dist_test.sh custom_config/spurious/knn.py foo 2
#bash ./tools/ood_dist_test_auto.sh custom_config/spurious/resnet18.py foo 2 --tab_name spurous


## id acc
#bash ./tools/dist_test.sh custom_config/id_acc/resnet50_topk.py /data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth 2 --metrics accuracy
#bash ./tools/dist_test_auto.sh custom_config/id_acc/resnet50_topk.py /data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth 2 --metrics accuracy
#bash ./tools/dist_test_auto.sh custom_config/id_acc/resnet50_topk.py /data/csxjiang/ood_ckpt/mmcls_offical/resnet34_8xb32_in1k_20210831-f257d4e6.pth 2 --metrics accuracy
#bash ./tools/dist_test_auto.sh custom_config/id_acc/resnet50_topk.py /data/csxjiang/ood_ckpt/mmcls_offical/resnet101_8xb32_in1k_20210831-539c63f8.pth 2 --metrics accuracy
