# a baseline method

import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from data_sampler import MyDistributedSampler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar-10 distributed Training')
parser.add_argument('data', default='dataset/cifar10', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--log-number', default='v999', help='the id for logging')

parser.add_argument('--train-layer', default=3, type=int,
                    help='layers to train in a pretrained model. '
                         '1 for last layer. 2 for last two layers, '
                         '3 for last three layers, 4 for the whole layers')

best_acc1 = 0

# important parameters, to spilt dataset unevenly to three nodes
batch_sizes = [312, 80, 300]


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    main_worker(args)


def main_worker(args):
    global best_acc1

    if args.distributed:
        print('Node ' + str(args.rank) + ' waiting other nodes to initialize...')
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print('Group initialization finish')
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=10)
        model_dict = model.state_dict()
        pretrained_dict = {k: v
                           for k, v in models.__dict__[args.arch](pretrained=True).state_dict().items()
                           if k.find('classifier.6') == -1}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        ct = 0
        for child in model.children():
            ct += 1
            if ct <= 2 and args.train_layer != 4:  # fraze all the feature layers
                for i, param in enumerate(child.parameters()):
                    param.requires_grad = False
                    print('froze', i, 'th feature layers.')
            else:  # fraze some classifier layers
                for i, param in enumerate(child.parameters()):
                    if i // 2 + args.train_layer <= 2 and args.train_layer != 4:
                        print('froze', i, 'th classifier layer')
                        param.requires_grad = False
        print('=| Finish loading pre-trained model "{}"'.format(args.arch))

    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=10)
        # ct = 0
        # for child in model.children():
        #     ct += 1
        #     if ct <= 2 and args.train_layer != 4:  # fraze all the feature layers
        #         for i, param in enumerate(child.parameters()):
        #             param.requires_grad = False
        #             print('froze', i, 'th feature layers.')
        #     else:  # fraze some classifier layers
        #         for i, param in enumerate(child.parameters()):
        #             if i // 2 + args.train_layer <= 2 and args.train_layer != 4:
        #                 print('froze', i, 'th classifier layer')
        #                 param.requires_grad = False
        print("=| Finish creating model '{}'".format(args.arch))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
        # print(model, 'Distributed model.')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print('Finish loading checkpoint')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    print("=> creating data loader ...")
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.49, 0.48, 0.45],
                                     std=[0.25, 0.24, 0.26])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.Resize(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        # transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if args.distributed:
        train_sampler = MyDistributedSampler(train_dataset, partition=batch_sizes)
        val_sampler = MyDistributedSampler(val_dataset, partition=batch_sizes)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_sizes[args.rank], shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_sizes[args.rank], shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    print("=| data loader created")

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    print('=> start training')
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1, train_batch_time, train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        write_log('log/train_epoch_log_' + args.log_number, epoch, train_loss, top1, train_batch_time)

        # evaluate on validation set
        acc1, val_batch_time, val_losses = validate(val_loader, model, criterion, args)
        write_log('log/test_log_' + args.log_number, epoch, val_losses, acc1, val_batch_time)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        epoch_end = time.time()
        print('epoch time:', epoch_end - epoch_start)
        with open('log/epoch_time_' + args.log_number, 'a') as f:
            f.write(str(epoch) + ' ' + str(epoch_end - epoch_start) + ' \n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    idle_time = AverageMeter('Idle', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        idle_time.update(time.time() - end)
        batch_start_time = time.time()
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        print('Node[' + str(args.rank) + '] finish batch[' + str(i) + '] forwarding')
        idle_start = time.time()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_start = time.time()
        loss.backward()
        loss_end = time.time()
        optimizer.step()
        print('forward:{:.2f}. backward:{:.2f}. step:{:.2f}'.format(
            loss_start - batch_start_time,
              loss_end - loss_start,
              time.time() - loss_end))

        # measure elapsed time
        elasped_time = time.time() - end
        batch_time.update(elasped_time)
        end = time.time()
        # print('batch time', elasped_time)
        write_log('log/train_log_' + args.log_number,
                  epoch * len(train_loader) + i, loss.item(), acc1[0].item(), elasped_time)

        # if i % args.print_freq == 0:
        progress.display(i)
    return top1.avg, batch_time.avg, losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, batch_time.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def write_log(filename, step, loss, acc, eplased_time):
    content = str(step) + ' ' + str(loss) + ' ' + str(acc) + ' ' + str(eplased_time) + ' \n'
    with open(filename, 'a') as f:
        f.write(content)


if __name__ == '__main__':
    main()
