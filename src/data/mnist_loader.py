import numpy as np
import struct
from array import array


class MnistDataloader:
    """
    Loads MNIST dataset from IDX binary files.
    Responsible ONLY for reading and returning data.
    """

    def __init__(
        self,
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def _read_images_labels(self, images_filepath, labels_filepath):
        """ Reads images and labels from IDX files """
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Label file magic number mismatch: {magic}")
            labels = array("B", file.read())

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Image file magic number mismatch: {magic}")
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(
                image_data[i * rows * cols:(i + 1) * rows * cols]
            ).reshape(rows, cols)
            images.append(img)

        return images, labels

    def load_data(self):
        """ Loads and returns the MNIST dataset as tuples:
            (x_train, y_train), (x_test, y_test)"""
        x_train, y_train = self._read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath
        )
        x_test, y_test = self._read_images_labels(
            self.test_images_filepath,
            self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)
