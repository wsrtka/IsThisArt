"""Script used for model training."""

import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from datasets import load_dataset, list_datasets


DATASET_NAME = 'keremberke/painting-style-classification'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    # check dataset availability
    assert DATASET_NAME in list_datasets(), f'{DATASET_NAME} is not a valid dataset name!'
    # download dataset
    ds = load_dataset(DATASET_NAME, name='full')
    ds.with_format('torch')
    train_ds = ds['train']
    val_ds = ds['val']
    test_ds = ds['test']
