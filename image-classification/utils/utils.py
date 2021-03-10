""" helper function

author baiyu
"""
import datetime
import functools
import json
import os
import re
import shutil
import time

import numpy
import torch.utils.data


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0]
                           for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1]
                           for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2]
                           for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(
        os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(
        re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(
        regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(
        re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def time_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return_val = func(*args, **kwargs)
        end = time.perf_counter()
        print('run {0:<1} spend : {1:.6f} sec'.format(
            func.__name__, end-start))
        return func_return_val
    return wrapper


def get_job_names(path='./datasets/ketmac'):
    return os.listdir(relat2abs_path(path))


def relat2abs_path(path: str) -> str:
    current_path = '/workspace/yuan-algorithm/image-classification/'
    # current_path = os.getcwd()  # '/workspace/yuan-algorithm/image-classification'
    print('current_path', current_path)
    return path.replace('./', current_path)


def abs2relat_path(path: str) -> str:
    current_path = os.getcwd()  # '/workspace/yuan-algorithm/image-classification'
    return path.replace(current_path, '.')


def make_result_path(result_path):
    os.makedirs(result_path, exist_ok=True)
    result_img_path = os.path.join(result_path, 'img')
    result_ref_path = os.path.join(result_path, 'ref')
    os.makedirs(result_img_path, exist_ok=True)
    os.makedirs(result_ref_path, exist_ok=True)
    return result_img_path, result_ref_path


def get_config(path):
    with open(path) as f:
        model = json.loads(f.read())
    return model


def get_folder_struct(img_dir: str, job_name: str):
    stru = img_dir.split('/')
    s = stru.index(job_name)

    stru = stru[s:]
    assert len(
        stru) == 4, "don't found  job_name / layer_name / lot_name / pannel_id / "
    # job_name / layer_name / lot_name / pannel_id /
    return stru


def delta_time():
    return datetime.timedelta(hours=8)


def copy_img_list(src_path, des_path):
    ret_ins = list()
    ret_ref = list()
    for (root, _, files) in os.walk(src_path):
        for f in files:
            if f.endswith('_ins.bmp'):
                img_path = os.path.join(des_path, 'img')
                ins = os.path.join(root, f)
                ret_ins.append(ins)
                des = ins.replace(src_path, img_path)
                os.makedirs(os.path.split(des)[0], exist_ok=True)
                # shutil.copy(ins, des)
                shutil.move(ins, des)

            elif f.endswith('_ref.bmp'):
                ref_path = os.path.join(des_path, 'ref')

                ref = os.path.join(root, f)
                ret_ref.append(ref)
                des = ref.replace(src_path, ref_path)
                os.makedirs(os.path.split(des)[0], exist_ok=True)
                # shutil.copy(ref, des)
                shutil.move(ref, des)

    return ret_ins, ret_ref


def mount2local(src_path, des_path):
    os.makedirs(des_path, exist_ok=True)
    img_path = os.path.join(des_path, 'img')
    ref_path = os.path.join(des_path, 'ref')
    print(img_path)
    if not os.path.exists(img_path):
        print('***************copy image file****************')
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(ref_path, exist_ok=True)

    ret_ins, ret_ref = copy_img_list(src_path, des_path)
    print('ret_ins, ret_ref ', len(ret_ins), len(ret_ref))
    assert len(ret_ins) == len(
        ret_ref), f'please check number of {ret_ref} and {ret_ins}'

    return img_path, ref_path


def get_pannel_folders(path):
    ret = set()
    for root, dirname, files in os.walk(path):
        for f in files:
            ret.add(root)
            break
    return sorted(list(ret))  # using order


def make_labels_path(labels: list, img_path, ret_path):

    for l in labels:
        os.makedirs(os.path.join(img_path, l), exist_ok=True)
        os.makedirs(os.path.join(ret_path, l), exist_ok=True)
