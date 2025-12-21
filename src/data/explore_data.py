import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def dataset_summary(x, y, name="Dataset"):
    """
    Prints basic dataset information.
    """
    print(f"--- {name} Summary ---")
    print(f"Number of samples: {len(x)}")
    print(f"Image shape: {x[0].shape}")
    print(f"Label type: {type(y[0])}")
    print()


def class_distribution(y):
    """
    Returns class distribution as a Counter.
    """
    return Counter(y)


def pixel_statistics(x):
    """
    Computes pixel-level statistics.
    """
    pixels = np.stack(x)
    print("--- Pixel Statistics ---")
    print(f"Min:  {pixels.min()}")
    print(f"Max:  {pixels.max()}")
    print(f"Mean: {pixels.mean():.4f}")
    print(f"Std:  {pixels.std():.4f}")
    print()


def visualize_samples(x, y, n=10):
    """
    Displays sample images with labels.
    """
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        axes[i].imshow(x[i], cmap="gray")
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y):
    """
    Plots class frequency histogram.
    """
    counter = Counter(y)
    classes = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.xlabel("Digit")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.show()
