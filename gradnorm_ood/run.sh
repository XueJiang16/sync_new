# list_method=( 'MSP' 'ODIN' 'Energy' 'new' 'GradNorm')
list_method=('MSP')

#list_dataset=('SUN' 'Places' 'Textures' 'iNaturalist')
 list_dataset=('Textures')
# # list_ckpt=('mobile_LT_a8/epoch_600' 'resnet152_LT_a8/epoch_100' 'resnet50_LT_a8/epoch_100')
#for method in ${list_method[*]}
#do
#for dataset in ${list_dataset[*]}
#do
# for i in {1,2,3,4,5,6,7,8,9,10}
# do
#bash ./custom_test_v3.sh $method $dataset /data/csxjiang/ood_ckpt/ood_ckpt_other/resnet101_batch256_imagenet.pth \
#    checkpoint_balanced/resnet101/ /data/csxjiang/meta/train_labeled.txt 0
# ./custom_test_v4.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_90.pth \
#     checkpoint0524/inat_res101_90/$method /mapai/haowenguo/data/ood_data/inat/train2018.json 0 
# ./custom_test_v4.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_95.pth \
#     checkpoint0524/inat_res101_95/$method /mapai/haowenguo/data/ood_data/inat/train2018.json 0 
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet152_LT_a8/epoch_100.pth \
#     checkpoint0518/resnet50/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 8 \
# ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
#     checkpoint0517/resnet101_purecosine/balanced  /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a2/epoch_100.pth \
#  checkpoint0521/a2_repeat${i}/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a2.txt 0
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0516/resnet101_balanced/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# done
#done
#done
#export CUDA_VISIBLE_DEVICES=1
#bash ./scripts/tune_mahalanobis.sh ./ood_ckpt/mmcls_offical/resnet101_8xb32_in1k_20210831-539c63f8.pth \
#    ./maha_ckpts /data/csxjiang/meta/train_labeled.txt
bash ./scripts/tune_mahalanobis.sh ./ood_ckpt/ood_ckpt_other/LT_a8/epoch_100.pth \
    ./maha_ckpts/test /data/csxjiang/meta/train_labeled_1percent.txt
#for dataset in ${list_dataset[*]}
#do
#bash  ./custom_test_v3_.sh Mahalanobis $dataset ./ood_ckpt/ood_ckpt_other/LT_a8/epoch_100.pth \
#      checkpoint0801/test_maha /data/csxjiang/meta/train_LT_a8.txt 0 \
#      --mahalanobis_param_path ./maha_ckpts/test/tune_mahalanobis/
#done
# for method in ${list_method[*]}
# do
# for dataset in ${list_dataset[*]}
# do
# for i in {1,2,3,4,5,6,7,8,9,10}
# do
# # ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_100.pth \
# #     checkpoint0512/inat_res101_plus2/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet50_LT_a8/epoch_100.pth \
# #     checkpoint0518/resnet50/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 8 \
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0517/resnet101_purecosine/balanced  /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth \
#  checkpoint0518/a8_repeat${i}/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0516/resnet101_balanced/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# done
# done
# done


# for method in ${list_method[*]}
# do
# for dataset in ${list_dataset[*]}
# do
# for i in {2,3,4,5,6,7,8}
# do
# # ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_100.pth \
# #     checkpoint0512/inat_res101_plus2/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/LT_a${i}/epoch_100.pth \
#     checkpoint0518/resnet101/LT_a$i/ /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a${i}.txt ${i} \
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0517/resnet101_purecosine/balanced  /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# # ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/code/SPL/jx/mmclassification/ckpt/LT_repeat${i}_a8/epoch_100.pth \
# #  checkpoint0429/a8_repeat${i}/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat${i}_a8.txt 0
# # ./custom_test_v3.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0516/resnet101_balanced/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# done
# done
# done

# for method in ${list_method[*]}
# do
# for dataset in ${list_dataset[*]}
# do
# # for i in {3,6,9}
# # do
# # ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/resnet101_imagnet10%_100e.pth \
# #     checkpoint0501/imagnet10percent/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt 0 \
# #     --mahalanobis_param_path /mapai/haowenguo/code/SPL/jx/gradnorm_ood/ckpt_mahalanobis/LT_a8/tune_mahalanobis
# # ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/inat/epoch_100.pth \
# #     checkpoint0512/inat_res101_plus2/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 0 \
# ./custom_test_v3_.sh $method $dataset /mapai/haowenguo/ckpt/ood_ckpt/ckpt/mobile_LT_a8/epoch_600.pth \
#     checkpoint0518/mobile/$method /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a8.txt 8 \
# # done
# done
# done