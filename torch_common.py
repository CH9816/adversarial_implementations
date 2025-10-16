import torch, torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


import torchvision.transforms as transforms

from random import randint

cpu, gpu = torch.device("cpu"), torch.device("cuda")


def get_datapoint(loader : DataLoader, labelled = True, device = cpu):
    #N = len(loader)
    #n = randint(0, N - 1)

    datapoint = next(iter(loader))

    return datapoint[0].to(device) if labelled else datapoint.to(device)



def topil(tensor : Tensor):
    #tensor = ((tensor + 1.) / 2.)
    tensor = tensor.cpu().detach()
    return transforms.ToPILImage()(
        tensor if len(list(tensor.shape)) == 3 else
        make_grid([img for img in tensor])
    )

def showtensor(x):
    topil(x.cpu()).show()







