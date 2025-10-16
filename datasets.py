import torch
from torch.utils.data import DataLoader, Subset
from torch import nn

import torchvision
from torchvision.datasets import MNIST, ImageNet, ImageFolder
from torchvision import transforms


cpu, gpu = torch.device("cpu"), torch.device("cuda")


def getTforms(size, range = [-1, 1]): 
    return transforms.Compose([    
        transforms.Resize((size, size)), transforms.ToTensor(), 
        # between -1 and 1 as per noise 
        transforms.Normalize(mean=[0.5], std=[0.5]) \
            if range == [-1, 1] else nn.Identity()
    ]) 


def get_train_test_split(dataset, split, bsize = 32, doShuffle = True):
    if split is None:
        return DataLoader(dataset, bsize, doShuffle)

    size = len(dataset)
    testSize = split * size
    trainSize = size - testSize

    return [
        # the dataLOADER created from the dataSET returned by .. 
        DataLoader(dataSet, bsize, doShuffle) 
            for dataSet in 

                # .. randomly splitting the whole dataset
                torch.utils.data.random_split(
                                    dataset, 
                                    [int(trainSize), int(testSize)])
    ]





def getImageNet(
        dir = "imagenet",
        targetImgSize = 256, bsize = 32,
        shuffle = True,
        train_test_split = None
    ):

    imageNet = ImageFolder(dir, getTforms(targetImgSize, [0, 1]))

    return get_train_test_split(imageNet, train_test_split, bsize, shuffle)









def getMNIST(
        dir = "mnist", 
        targetImgSize = 28, bsize = 32,
        # download if dataset is not already downloaded 
        # in folder specified by dir
        downloadIfNotExists = False,
        shuffle = True,
        train_test_split = None
    ):

    pass;


    mnist = MNIST(dir, True, getTforms(targetImgSize, [0, 1]), download=downloadIfNotExists)

    return get_train_test_split(mnist, train_test_split, bsize, shuffle)

    #if train_test_split is None:

    #    dataset = DataLoader(mnist, bsize, shuffle)
    #    return dataset;

    #else:
    #    size = len(mnist)
    #    test = train_test_split * size
    #    #train = (1 - train_test_split) * size
    #    train = size - test
    #    return [
    #                DataLoader(x, bsize, shuffle) for x in 
    #                    torch.utils.data.random_split(mnist, [int(train), int(test)])
    #           ]