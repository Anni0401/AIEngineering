from os.path import join
import sys
import os
from pathlib import Path
from src.data.mnist_loader import MnistDataloader
from src.data.explore_data import (
    dataset_summary,
    class_distribution,
    pixel_statistics
)

input_path  = os.path.join(os.path.dirname(__file__), '../../data')


mnist = MnistDataloader(
    join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"),
    join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
    join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"),
    join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
