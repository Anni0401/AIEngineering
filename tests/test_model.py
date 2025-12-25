from os.path import join
import sys 
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import numpy as np
import torch
from src.models.cnn import MNISTCNN
def test_model_forward():
    model = MNISTCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)

    assert out.shape == (4, 10)
