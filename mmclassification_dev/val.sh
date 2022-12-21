export PICK_CLASS=0

#bash ./tools/ood_dist_test.sh custom_config/validation/resnet50.py foo 2


bash ./tools/ood_dist_test_auto.sh custom_config/validation/resnet50.py foo 2 --tab_name val_set
