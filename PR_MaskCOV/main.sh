# debug
#python train.py --data CUB --use_maskcov --epoch 2 --tnw 0 --vnw 0 --backbone resnet50 --cls_2 --swap_num 2 2 --cp 2 --detail cub_model1 --mask_num 1

CUDA_VISIBLE_DEVICES=0,1 python train.py --data COTTON --use_maskcov --epoch 160 --backbone resnet50 --cls_2 --detail cub_model1 --lr_scheduler stage

CUDA_VISIBLE_DEVICES=0,1 python test.py --data COTTON --backbone resnet50  --save ./logs/COTTON/cub_model1_2*2/savepoint-last-model-steps2400-valtop1acc0.1917_valtop3acc0.3708.pth




# Attention Please
#1. change "COTTON, Soybean200, Soybean2000, soybean_gene, R1, R3, R4, R5, R6" the same to paper , 46 line train.py and config.py file ; test.py 33 line