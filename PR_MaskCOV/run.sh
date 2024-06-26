CUDA_VISIBLE_DEVICES=0,1 python train.py --data soybean_gene --use_maskcov --epoch 240 --backbone resnet50 --cls_2 --detail ori_cos_step --tb 16 --lr_scheduler cos_step
CUDA_VISIBLE_DEVICES=0,1 python train_custom.py --data soybean_gene --use_maskcov --epoch 240 --backbone resnet50 --cls_2 --detail dsa1_cos_step --tb 16 --lr_scheduler cos_step --method 1
CUDA_VISIBLE_DEVICES=0,1 python train_custom.py --data soybean_gene --use_maskcov --epoch 240 --backbone resnet50 --cls_2 --detail dsa2_cos_step --tb 16 --lr_scheduler cos_step --method 2


CUDA_VISIBLE_DEVICES=2 python train_custom.py --data soybean_gene --use_maskcov --epoch 160 --backbone resnet50 --cls_2 --detail dsa0_cos_step --tb 16 --lr_scheduler cos_step --method 0 --small

CUDA_VISIBLE_DEVICES=0,1 python train_custom.py --data soybean_gene --use_maskcov --epoch 160 --backbone resnet50 --cls_2 --detail dsa1_cos_step --tb 16 --lr_scheduler cos_step --method 1