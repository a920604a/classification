# test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import argparse
import math
import os
import shutil

import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from conf import settings
from utils.loader import (get_test_dataloader, get_training_dataloader,
                          get_valid_dataloader)
from utils.network import get_network


def parse_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True,
                        help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true',
                        default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16,
                        help='batch size for dataloader')
    parser.add_argument('-s', action="store_true",
                        help='save result')
    parser.add_argument('-src', type=str, default='datasets/val')
    parser.add_argument('-des', type=str, default='result')
    parser.add_argument('-ts', type=float, default=0.0)

    return parser.parse_args()


def filter_ts(code_convert, pre, pro, ts=0.7):
    if code_convert[pre] == 'OK':
        if pro[pre] > ts:
            pre = 1
        else:
            pre = 0
    return pre


def evaluate(args, net, valid_loader):
    code_convert = {v: k for k, v in valid_loader.dataset.class_to_idx.items()}
    if args.des and args.s:
        [os.makedirs(os.path.join(args.des, folder), exist_ok=True)
         for folder in valid_loader.dataset.classes]

    correct_1 = 0.0
    total = 0
    y_true = []
    y_pred = []
    bz = valid_loader.batch_size
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(valid_loader):

            batch_img_path = [
                f for f, _ in valid_loader.dataset.samples[n_iter*bz:(1+n_iter)*bz]]
            print("iteration: {}\ttotal {} iterations".format(
                n_iter, len(valid_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')

            output = net(image)
            probs = torch.sigmoid(output)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)

            prediction = []
            for l, p, m, b in zip(label, pred, batch_img_path, probs):
                lab = int(l.cpu().numpy())
                pre = int(p.cpu().numpy())
                pro = b.cpu().numpy()

                if args.ts > 0.5:
                    pre = filter_ts(code_convert, pre, pro, ts=args.ts)

                img_name = os.path.split(m)[1]
                # if lab == pre:
                print(lab, pre)
                y_true.append(lab)
                y_pred.append(pre)
                if args.s and args.des:
                    des_path = os.path.join(
                        args.des, code_convert[lab], code_convert[pre])
                    os.makedirs(des_path, exist_ok=True)
                    shutil.copy(m, os.path.join(des_path,  img_name))

                prediction.append([pre])
            prediction = torch.cuda.FloatTensor(prediction)
            correct = prediction.eq(label).float()
            # compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print("Top 1 err: ", 1 - correct_1 / len(valid_loader.dataset))
    print("Top 1 accuracy: ", 100*correct_1 / len(valid_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel()
    #                                          for p in net.parameters())))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)


    for i,(cc,c) in enumerate(zip(code_convert.values(),cm)):
        print('{} precision :{:.4f} %'.format(cc , 100*c[i]/sum(c)))


def predict(args, net, test_loader, code_convert):
    bz = test_loader.batch_size
    with torch.no_grad():
        for n_iter, (image, _) in enumerate(test_loader):
            batch_img_path = [
                f for f, _ in test_loader.dataset.samples[n_iter*bz:(1+n_iter)*bz]]
            print("iteration: {}\ttotal {} iterations".format(
                n_iter, len(test_loader)))

            if args.gpu:
                image = image.cuda()

            output = net(image)
            _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)

            for p, m in zip(pred, batch_img_path):
                pre = int(p.cpu().numpy())
                img_name = os.path.split(m)[1]
                if args.s and args.des:
                    des_path = os.path.join(
                        args.des,  code_convert[pre])
                    os.makedirs(des_path, exist_ok=True)
                    shutil.copy(m, os.path.join(des_path,  img_name))

            # compute top1

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')


if __name__ == '__main__':

    args = parse_default_args()
    net = get_network(args)

    valid_loader = get_valid_dataloader(
        root_path=args.src,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        size=(128, 128)
    )

    test_loader = get_valid_dataloader(
        root_path=args.src,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        size=(128, 128)
    )

    net.load_state_dict(torch.load(args.weights))
    # print(net)
    net.eval()

    evaluate(args, net, valid_loader)
    code_convert = {0:"NG",1:"OK"}
    predict(args, net, test_loader,code_convert)
