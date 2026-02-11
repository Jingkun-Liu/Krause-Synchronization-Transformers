import torch
import numpy as np
import random
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_imagenet_loaders(data_path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = datasets.ImageNet(root=data_path, split='train')
    testset = datasets.ImageNet(root=data_path, split='val')

    trainset.transform = transform_train
    testset.transform = transform_test

    train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
    test_sampler = DistributedSampler(testset, shuffle=False)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=48,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return train_loader, test_loader, trainset, testset, train_sampler, test_sampler

