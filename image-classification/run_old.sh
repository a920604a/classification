#! /bin/bash
# python3 train.py -net xception -gpu -b 4 -lr 0.001 -optim adamw 
# python3 test.py -net xception -weights ./checkpoint/xception/Wednesday_27_January_2021_08h_21m_21s/xception-122-best.pth  -gpu -b 4
# Top 1 accuracy:  tensor(90.8815, device='cuda:0')
# [[292  37]
#  [ 23 306]]

# python3 train.py -net xception -gpu -b 4 -lr 0.0001 -optim adamw 
# python3 test.py -net xception -weights ./checkpoint/xception/Thursday_28_January_2021_09h_25m_50s/xception-133-best.pth  -gpu -b 4
# Top 1 accuracy:  tensor(88.1459, device='cuda:0')
# [[277  52]
#  [ 26 303]]

# + transforms.RandomVerticalFlip(),
# - transforms.RandomRotation(15),
# python3 train.py -net xception -gpu -b 4 -lr 0.001 -optim adamw 
# python3 test.py -net xception -weights ./checkpoint/xception/Friday_29_January_2021_02h_09m_33s/xception-154-best.pth  -gpu -b 4
# Top 1 accuracy:  tensor(89.9696, device='cuda:0')
# [[292  37]
#  [ 29 300]]

# python3 train.py -net resnext101 -gpu -b 2 -lr 0.001 -optim adamw 
# python3 test.py -net resnext101 -weights ./checkpoint/resnext101/Friday_29_January_2021_22h_59m_23s/resnext101-177-best.pth  -gpu -b 2
# Top 1 accuracy:  tensor(89.5137, device='cuda:0')
# [[282  47]
#  [ 22 307]]

# python3 train.py -net attention92 -gpu -b 4 -lr 0.001 -optim adamw 
# python3 test.py -net attention92 -weights ./checkpoint/attention92/Sunday_31_January_2021_09h_12m_45s/attention92-136-best.pth \
#             -gpu -b 4 -s -src ./datasets/val -des ./result
# Top 1 accuracy:  tensor(90.8815, device='cuda:0')
# [[292  37]
#  [ 23 306]]

# python3 train.py -net xception -gpu -b 4 -lr 0.001 -optim adam

# b=4 , attention92 , resnet101 , resnet101
# b=2 resnext101 , nasnet
# out of memory : inceptionv4
# for net in nasnet
# do 
#     python3 train.py -net $net -gpu -b 2 -lr 0.001 -optim adamw  
# done



