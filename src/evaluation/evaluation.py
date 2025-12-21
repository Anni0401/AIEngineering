import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.cnn import MNISTCNN


def evaluate_model(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "classification_report": classification_report(all_labels, all_preds, output_dict=True),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "predictions": all_preds,
        "labels": all_labels,
    }

    return results
