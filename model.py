import jittor as jt
from jittor import nn, init
#from jittor.autograd import Variable?
#from jittor.optim import lr_scheduler ?
import jittor.nn as F


# three branch network

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, norm_layer=nn.BatchNorm2d, nonlinear='relu', pooling='max'):
        super(Block, self).__init__()
        conc_block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)]

        conc_block.append(norm_layer(out_channels))

        if nonlinear == 'elu':
            conc_block.append(nn.ELU(True))
        elif nonlinear == 'relu':
            conc_block.append(nn.ReLU(True))
        elif nonlinear == 'leaky_relu':
            conc_block.append(nn.LeakyReLU(0.2, True))

        if pooling is not None:
            conc_block.append(nn.Pool(kernel_size=3, stride=2, padding=0, op='maximum'))
#nn.MaxPool2d is replaced with nn.Pool
        self.model = nn.Sequential(*conc_block)

    def execute(self, x):
        out = self.model(x)
        return out


class Part_f(nn.Module):
    def __init__(self, input_channel=1, is_code=True):
        super(Part_f, self).__init__()
        # if using self-coding module
        if is_code:
            part_f = [nn.Conv2d(in_channels=input_channel, out_channels=1, kernel_size=1, stride=1, bias=False)]
            part_f += [nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)]
        else:
            part_f = [nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)]
        part_f += [nn.BatchNorm2d(8)]
        self.model = nn.Sequential(*part_f)

    def execute(self, x):
        out = self.model(x) #self.model(x) is replaced with self.model.execute(x)
        return out

class ScNet(nn.Module):
    def __init__(self, input_nc=6, ndf=5, nonlinear='relu'):
        super(ScNet, self).__init__()

        self.input_nc = input_nc
        # the branch of ScNet
        self.branch0 = nn.Sequential(Part_f(3, is_code=True), Block(8, 8, pooling=None), Block(8, 8, pooling=None))
        self.branch1 = nn.Sequential(Part_f(3, is_code=True), Block(8, 8, pooling=None), Block(8, 8, pooling=None))
        self.branch2 = nn.Sequential(Part_f(3, is_code=True), Block(8, 8, pooling=None), Block(8, 8, pooling=None))

        # the basic of ScNet
        self.basic = nn.Sequential(
            Block(in_channels=24, out_channels=2**ndf, nonlinear=nonlinear),
            Block(in_channels=2**ndf, out_channels=2**(ndf + 0), nonlinear=nonlinear),
            Block(in_channels=2**(ndf + 0), out_channels=2**(ndf + 1), nonlinear=nonlinear),
            Block(in_channels=2**(ndf + 1), out_channels=2**(ndf + 2), nonlinear=nonlinear),
            Block(in_channels=2**(ndf + 2), out_channels=2**(ndf + 3), nonlinear=nonlinear),
            Block(in_channels=2**(ndf + 3), out_channels=2**(ndf + 3), nonlinear=nonlinear, pooling=None)
        )

        self.fc = nn.Linear(256, 2)

    def execute(self, x):
        y1 = self.branch0(x)
        y2 = self.branch1(x)
        y3 = self.branch2(x)
        x = jt.contrib.concat((y1, y2, y3), 1) #jt.contrib.concat is used instead of torch.cat
        x = self.basic(x)

        x = jt.nn.adaptive_avg_pool2d(x, (1, 1)) #jt.nn.adaptive_avg_pool2d is used instead of F.adaptive_avg_pool2d
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out