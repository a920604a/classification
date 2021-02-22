#! /bin/bash

python3 train.py -net xception -gpu -b 4 -lr 0.003 -optim adam 
python3 test.py -net attention92 -weights ./checkpoint/attention92/Sunday_31_January_2021_09h_12m_45s/attention92-136-best.pth \
             -gpu -b 4 -s -src ./datasets/val -des ./result
