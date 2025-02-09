# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from criterion import LSR
from utils.loader import get_training_dataloader, get_valid_dataloader
from utils.network import get_network
from utils.tricks import WarmUpLR, init_weights, split_weights
from utils.utils import (best_acc_weights, last_epoch, most_recent_folder,
                         most_recent_weights, time_wrapper)


@time_wrapper
def train(epoch):

    # start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar(
                    'LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar(
                    'LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    # finish = time.time()

    # print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    valid_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in valid_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        valid_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        valid_loss / len(valid_loader.dataset),
        correct.float() / len(valid_loader.dataset),
        finish - start
    ))
    print()

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Validation/Average loss',
                          valid_loss / len(valid_loader.dataset), epoch)
        writer.add_scalar('Validation/Accuracy',
                          correct.float() / len(valid_loader.dataset), epoch)

    return correct.float() / len(valid_loader.dataset)


def parse_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true',
                        default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128,
                        help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1,
                        help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('-resume', action='store_true',
                        default=False, help='resume training')
    parser.add_argument('-optim', type=str, required=False, help='optimizer')

    parser.add_argument('--train_path', type=str,
                        default='datasets/train-0.8',)
    parser.add_argument('--valid_path', type=str,
                        default='datasets/val-0.2')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_default_args()
    net = get_network(args)

    # xavier init
    net = init_weights(net)

    # data preprocessing:
    training_loader = get_training_dataloader(
        root_path=args.train_path,
        num_workers=4,
        batch_size=args.b,
        shuffle=True,
        size=(128, 128)
    )

    valid_loader = get_valid_dataloader(
        root_path=args.valid_path,
        num_workers=4,
        batch_size=args.b,
        shuffle=False,
        size=(128, 128)
    )

    loss_function = nn.CrossEntropyLoss()
    # loss_function = LSR()

    # apply no weight decay on bias
    params = split_weights(net)

    if args.optim == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr)
        train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(
            training_loader), epochs=settings.EPOCH)  # add by james
    else:
        if args.optim == 'adam':
            optimizer = optim.Adam(params, lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

        elif args.optim == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=settings.MILESTONES, gamma=0.2)  # learning rate decay

    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(
            settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    # since tensorboard can't overwrite old values
    # so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(
                settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to evaluate acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
