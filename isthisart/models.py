"""Module containing various deep learning models."""

import torch

from torch import nn


class TinyVGG(nn.Module):
    """
    TinyVGG model based on:
    https://poloclub.github.io/cnn-explainer/
    """
