#!/usr/bin/env python
# encoding: utf-8
'''
@author: lindq
@contact: lindq@shanghaitech.edu.cn
'''

from __future__ import print_function
import argparse
import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
from kdnn_cuda import QKNet
from PGD_attack import attack
# from kdnn import *

# Hyper-parameters
params = {
    'attack_type': 'pgd',
    'epsilon': 0.3,
    'k': 100,
    'step_size': 0.01
}

mean, std = 0.1307, 0.3081


def normalize(t):
    return (t - mean) / std


def un_normalize(t):
    return t*std + mean

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    count_batch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(normalize(data))
        # loss = F.nll_loss(output, target)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # np.save('center_dict.npy', model.center_dict)
            # np.save('table_dict.npy', model.table_dict)
        total_loss += loss.item()
        count_batch += 1
    print('Train Epoch: {} \t\t\t\tAverage Loss: {:.6f}'.format(
        epoch, total_loss / count_batch))


def test(args, model, device, test_loader, is_adv=False):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_adv:
                with torch.enable_grad():
                    adv = attack(model, criterion, normalize, data, target, params)
                    adv.requires_grad = False
                    # adv = normalize(adv)
                    data = adv.cuda()
            output = model(normalize(data))
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    parser.add_argument('--pretrained-model-dir', type=str,
                        default='models/mnist_cnn_moving_average_all.pt')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = QKNet(device=device).to(device)
    # filepath = './models/mnist_cnn_moving_average_all.pt'
    if os.path.isfile(args.pretrained_model_dir):
        state = torch.load(args.pretrained_model_dir)
        model.load_state_dict(state['state_dict'])
        model.set_center(state['center'], state['table'])
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    training_time = 0.
    best_test_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = timer()
        scheduler.step()
        train(args, model, device, train_loader, optimizer, epoch)
        delta_time = timer() - start
        training_time += delta_time
        print(' {:.2f}s/epoch'.format(delta_time))
        if epoch%args.eval_freq == 0:
            nat_test_acc = test(args, model, device, test_loader)
            adv_test_acc = test(args, model, device, test_loader, is_adv=True)
            if (args.save_model):
                if adv_test_acc > best_test_acc:
                    best_test_acc = adv_test_acc
                    state = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'center': model.center_dict,
                        'table': model.table_dict,
                        'clean_acc': nat_test_acc,
                        'robust_acc': adv_test_acc,
                    }
                    filepath = "./models/mnist_cnn_moving_average_all.pt"
                    torch.save(state, filepath)


if __name__ == '__main__':
    main()
