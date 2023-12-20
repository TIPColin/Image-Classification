from __future__ import print_function
import argparse
import os

# from torchvision.datasets import ImageFolder
import myDataset
import networks
import numpy as np
import jittor as jt
#import torch.backends.cudnn as cudnn: not needed, jittor automatically uses cuDNN optimizations
import jittor.nn as nn
import jittor.optim as optim
import jittor.transform as transforms
from PIL import Image
from model import ScNet

import jittor.models as models
from jittor import Var
from tqdm import tqdm


# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"

# Training settings
parser = argparse.ArgumentParser(description='Jittor NI vs CG')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--patch-size', type=int, default=96, metavar='N',
                    help='input the patch size of the network during training and testing (default: 96)')
parser.add_argument('--log-dir', default='/data/xshen/2ndproject/JCST/log',
                    help='folder to output model checkpoints')
parser.add_argument('--img_mode', type=str, default='RGB',
                    help='chooses how images are loaded')
parser.add_argument('--input_nc', type=int, default=3,
                    help='# of input image channels')
parser.add_argument('--epochs', type=int, default=1200, metavar='N',
                    help='number of epochs to train (default: 1200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--loss-adjust', default=400, type=int,
                    help='how many epochs to change the learning rate (default: 400)')
parser.add_argument('--summary-interval', type=int, default=50,
                    help='how many epochs to summary the log')

args = parser.parse_args()

args.cuda = not args.no_cuda and jt.flags.use_cuda
np.random.seed(args.seed)
if args.cuda:
    jt.flags.use_cuda = True  # Assign True to jt.flags.use_cuda to enable CUDA in Jittor.
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
# The path of data
data_root = '/data/xshen/partition'

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# Data loading code
# You need to refine this for your data set directory
train_dir = os.path.join(data_root, 'train')
val_dir = os.path.join(data_root, 'test')

normalize = transforms.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

'''train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(train_dir,
                        transforms.Compose([
                            transforms.RandomCrop(args.patch_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                        ])),
    batch_size=args.batch_size, shuffle=True, half_constraint=True,
    sampler_type='RandomBalancedSampler', **kwargs)

val_loader = myDataset.DataLoader(
    myDataset.MyDataset(val_dir,
                        transforms.Compose([
                            transforms.CenterCrop(args.patch_size),
                            transforms.ToTensor(),
                            normalize
                        ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
'''
train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(train_dir, 
                        transforms.Compose([transforms.RandomCrop(args.patch_size), 
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(), 
                                            normalize])),
                        shuffle=True, batch_size=args.batch_size, 
                        half_constraint=True, sampler_type='RandomBalancedSampler', 
                        drop_last=True, num_workers=0)

val_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(val_dir, 
                        transforms.Compose([transforms.CenterCrop(args.patch_size), 
                                            transforms.ToTensor(), normalize])),
                        shuffle=False, batch_size=args.test_batch_size, 
                        half_constraint=True, sampler_type='RandomBalancedSampler', 
                        drop_last=True, num_workers=0)

def main():
    # instantiate model and initialize weights
    model = ScNet()

    networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    if args.cuda:
        model.cuda()
    print('model load!')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.lr)

    for epoch in range(1, args.epochs + 1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, criterion, epoch)

        if epoch % args.summary_interval == 0:
            val_acc, val_loss = test(val_loader, model, criterion, epoch)


def train(train_loader, model, optimizer, criterion, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))

    running_loss = 0
    running_corrects = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data_var, target_var = jt.float32(data), jt.float32(target)

        prediction = model(data_var)

        _, preds = jt.argmax(prediction, 1)

        loss = criterion(prediction, target_var)

        # statistics
        running_loss += loss.item()
        running_corrects += jt.sum(preds == target_var.data).item()

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]   Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data_var), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

        if epoch % args.log_interval == 0:
            jt.save({'epoch': epoch,
                     'state_dict': model.state_dict()},
                    '{}/checkpoint_{}.pth'.format(args.log_dir, epoch))

    running_loss = running_loss / (len(train_loader.dataset) // args.batch_size)
    ave_corrects = 100. * running_corrects / len(train_loader.dataset)
    print('Train Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, running_loss, running_corrects, len(train_loader.dataset), ave_corrects))
    return ave_corrects, running_loss


def test(val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()

    test_loss = 0
    correct = 0

    pbar = tqdm(enumerate(val_loader))
    for batch_idx, (data, target) in pbar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = jt.float32(data).stop_grad(), jt.float32(target)

        # compute output
        output = model(data)
        test_loss += criterion(output, target).item()    # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]    # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss = test_loss / (len(val_loader.dataset) // args.test_batch_size)
    ave_correct = 100. * correct / len(val_loader.dataset)
    print('Test Epoch: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(val_loader.dataset), ave_correct))
    return ave_correct, test_loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.loss_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
