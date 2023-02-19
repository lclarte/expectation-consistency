import argparse
import datetime
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import wandb

from averagemeter import AverageMeter
from dataloader import get_dataloader
import resnet
import densenet


best_prec1 = 0

## 
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
## 

def get_model(model_str):
    if model_str == "cifar10_resnet20":
        return resnet.resnet20()
    if model_str == "cifar10_resnet32":
        return resnet.resnet32()
    if model_str == "cifar10_resnet44":
        return resnet.resnet32()
    if model_str == "cifar10_resnet56":
        return resnet.resnet56()
    if model_str == "cifar10_resnet110":
        return resnet.resnet110()
    if model_str == "cifar10_densenet121":
        return densenet.densenet121()
    if model_str == "cifar10_densenet169":
        return densenet.densenet169()
    if model_str == "cifar10_densenet201":
        return densenet.densenet201()
    else:
        raise Exception()
## 

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target#.cuda()
            input_var = input#.cuda()
            target_var = target#.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    wandb.log({
        "validation/accuracy" : top1.avg
    })

    return top1.avg

## 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_str', default='cifar10_resnet20')
parser.add_argument('--dataset_str', default='cifar10')
parser.add_argument('--save_dir', default='default_dir')
parser.add_argument('--train_ratio', default=0.9, type=float)
parser.add_argument('--random_seed', default=42, help='random seed for validation split', type=int)
parser.add_argument('--bool_train_model', default=True)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', default=0.1, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

def main(args):
    model_str = args.model_str
    dataset_str = args.dataset_str
    save_dir = args.save_dir
    train_ratio = args.train_ratio
    random_seed = args.random_seed
    bool_train_model = args.bool_train_model
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    
    # Check the save_dir exists or not

    if save_dir == 'default_dir':
        save_dir = f'{dataset_str}.{model_str}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb.init(project="expectation-consistency", entity="lucas-a-clarte", config = {
        "learning_rate": lr,
        "epochs": epochs,
        "dataset" : dataset_str,
        "architecture" : model_str,
        "random_seed" : random_seed,
        "train_ratio" : train_ratio
    })

    model = get_model(model_str)
    # NOTE : Usefulness of this line ? 
    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloader(dataset_str, random_seed, train_ratio)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum = momentum, weight_decay = weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch = -1)
    if model_str == "resnet110":
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    best_prec1 = 0

    for epoch in range(epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, 'checkpoint.th'))

        """
        save_checkpoint({
                'training_args' : args,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, 'model.th'))
        """
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.th'))


if __name__ == "__main__":
    main(parser.parse_args())