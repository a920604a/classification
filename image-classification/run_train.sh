#! /bin/bash
###
 # @Author: yuan
 # @Date: 2021-02-19 16:06:15
 # @LastEditTime: 2021-02-23 14:27:21
 # @FilePath: /image-classification/run_train.sh
### 

# python3 train.py -net xception -gpu -b 4 -lr 0.003 -optim adam 
python3 test.py -net xception -weights ./checkpoint/xception/Thursday_04_February_2021_22h_28m_22s/xception-132-best.pth \
            -gpu -b 4 -s -src ./datasets/val-0.2 -des ./result