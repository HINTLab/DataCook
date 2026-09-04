import mlconfig
import torch
import torch.nn as nn
import torchvision

from . import ResNet, vgg, convnext, ResNet3d, ResNet_224

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)

# Models
mlconfig.register(ResNet.ResNet)
mlconfig.register(ResNet.ResNet18)
mlconfig.register(ResNet.ResNet34)
mlconfig.register(ResNet.ResNet50)
mlconfig.register(ResNet.ResNet101)
mlconfig.register(ResNet.ResNet152)
mlconfig.register(vgg.VGG)
mlconfig.register(convnext.convnext_tiny)
mlconfig.register(ResNet3d.ResNet18_3d)
mlconfig.register(ResNet_224.ResNet18_224)

# CUDA Options
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

