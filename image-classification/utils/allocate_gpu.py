'''
Author: yuan
Date: 2021-02-23 14:30:41
LastEditTime: 2021-02-23 14:47:44
FilePath: /image-classification/utils/allocate_gpu.py
'''
import os
import torch


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    print(devices_info)
    return total, used


def occumpy_mem(cuda_device, per):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    
    if int(total*per) + used > total:
        print('out of memory')
    max_mem = int(total * per)
    block_mem = max_mem - used

    print('reserved mem:{} , already uesd :{} ,total mem:{}'.format(block_mem , used , total))
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x