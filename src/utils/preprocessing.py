import numpy as np
import random

def normalize_images(x):
    """
    Normalize pixel values from [0, 255] to [0, 1].
    """
    x = np.array(x, dtype=np.float32)
    return x / 255.0
def add_channel_dimension(x):
    """
    Converts (N, 28, 28) → (N, 1, 28, 28)
    """
    return np.expand_dims(x, axis=1)
def prepare_labels(y):
    """
    Converts labels to NumPy int64 array.
    """
    return np.array(y, dtype=np.int64)
def random_shift(image, max_shift=2):
    """
    Randomly shifts image horizontally and vertically.
    """
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    shifted = np.roll(image, dx, axis=0)
    shifted = np.roll(shifted, dy, axis=1)
    return shifted


def random_noise(image, noise_level=0.02):
    """
    Adds random Gaussian noise.
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)
def augment_dataset(x, y, augmentation_factor=1):
    """
    Augments dataset by creating new samples.
    """
    augmented_x = []
    augmented_y = []

    for img, label in zip(x, y):
        augmented_x.append(img)
        augmented_y.append(label)

        for _ in range(augmentation_factor):
            img_aug = random_shift(img)
            img_aug = random_noise(img_aug)
            augmented_x.append(img_aug)
            augmented_y.append(label)

    return np.array(augmented_x), np.array(augmented_y)
def preprocess_pipeline(x, y, augment=False):
    """
    Full preprocessing pipeline.
    """
    x = normalize_images(x)
    y = prepare_labels(y)

    if augment:
        x, y = augment_dataset(x, y)

    x = add_channel_dimension(x)
    return x, y

