import sys 
import os
from os.path import join
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from src.data.mnist_loader import MnistDataloader
import numpy as np
from src.utils.preprocessing import preprocess_pipeline

def test_preprocessing_output_shape():
    x = np.random.randint(0, 255, (10, 28, 28))
    y = np.random.randint(0, 10, 10)

    x_prep, y_prep = preprocess_pipeline(x, y, augment=False)

    assert x_prep.shape == (10, 1, 28, 28)
    assert y_prep.shape == (10,)
