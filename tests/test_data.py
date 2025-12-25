import sys 
import os
from os.path import join
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from src.data.mnist_loader import MnistDataloader
import numpy as np

def test_mnist_dataloader_loads_data():
    loader = MnistDataloader(
        training_images_filepath="data/train-images-idx3-ubyte/train-images-idx3-ubyte",
        training_labels_filepath="data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        test_images_filepath="data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        test_labels_filepath="data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
    )

    (x_train, y_train), (x_test, y_test) = loader.load_data()

   
    assert x_train is not None
    assert y_train is not None
    assert x_test is not None
    assert y_test is not None

   
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


    assert len(x_train) == 60000
    assert len(x_test) == 10000


    assert x_train[0].shape == (28, 28)
    assert x_test[0].shape == (28, 28)

  
    assert isinstance(x_train[0], np.ndarray)
    assert x_train[0].dtype == np.uint8


    assert 0 <= y_train[0] <= 9

