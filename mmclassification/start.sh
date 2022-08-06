# mkdir -p /apdcephfs/private_haowenguo/shared_info/haowenguo/$TJ_TASK_ID
# rm -rf ckpt
# ln -s /apdcephfs/private_haowenguo/shared_info/haowenguo/$TJ_TASK_ID ckpt
# pip install -e .
# bash ./tools/dist_train.sh custom_config/LT_a2.py 8
# bash ./tools/dist_train.sh custom_config/LT_a3.py 8
# bash ./tools/dist_train.sh custom_config/LT_a4.py 8
# bash ./tools/dist_train.sh custom_config/LT_a5.py 8
# bash ./tools/dist_train.sh custom_config/LT_a6.py 8
# bash ./tools/dist_train.sh custom_config/LT_a7.py 8
# bash ./tools/dist_train.sh custom_config/LT_a8.py 8
# bash ./tools/dist_train.sh custom_config/resnet101_imagenet_110.py 8 --resume-from /mapai/haowenguo/ckpt/ood_ckpt/resnet101_batch256_imagenet.pth
# bash ./tools/dist_train.sh custom_config/BLT_a2.py 8
# bash ./tools/dist_train.sh custom_config/BLT_a8.py 8
# for i in {1,2,4,5,7,8,9,10}
#for i in {1,2,3}
#do
#bash ./tools/dist_train.sh custom_config/LT_repeat${i}_a8.py 4
#done
# bash ./tools/dist_train.sh custom_config/mobilenet3_large.py 8
# bash ./tools/dist_train.sh custom_config/resnet152.py 8
 bash ./tools/dist_train.sh custom_config/resnet50.py 2
 bash ./tools/dist_train.sh custom_config/resnet50_ft_cbwd.py 2
#bash ./tools/dist_test.sh custom_config/resnet101_imagenet.py /data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth 2 --metrics accuracy
#bash ./tools/dist_train.sh custom_config/resnet101_imagenet.py 2
#bash ./tools/dist_train.sh custom_config/LT_a8.py 2
#bash ./tools/dist_train.sh custom_config/cifar.py 2