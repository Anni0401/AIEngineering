import torch
import numpy as np
from src.models.cnn import MNISTCNN


def load_model(model_path="mnist_cnn.pth", device="cpu"):
    """ Loads the trained model from disk. """
    model = MNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(image: np.ndarray, model):
    """
    image: numpy array of shape (28, 28), values [0, 255]
    """
    image = image / 255.0
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        pred = logits.argmax(dim=1).item()

    return pred
