import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn import MNISTCNN
from src.training.engine import train_one_epoch, evaluate
from src.utils.preprocessing import preprocess_pipeline
from src.data.mnist_loader import MnistDataloader
from os.path import join


def train():
    # -----------------------------
    # 1. Initialize W&B
    # -----------------------------
    wandb.init(project="my_project", config={
    "learning_rate": 0.0005,
    "batch_size": 32,
    "epochs": 10
    })
    config = wandb.config

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # 2. Load data
    # -----------------------------
    input_path = "data"

    mnist = MnistDataloader(
        join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte"),
        join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte"),
        join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"),
        join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"),
    )

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # -----------------------------
    # 3. Preprocessing
    # -----------------------------
    x_train, y_train = preprocess_pipeline(
        x_train, y_train, augment=True
    )
    x_test, y_test = preprocess_pipeline(
        x_test, y_test, augment=False
    )
    x_train = torch.from_numpy(x_train).float()  
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()  
    y_test = torch.from_numpy(y_test).long()
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # -----------------------------
    # 4. Model
    # -----------------------------
    model = MNISTCNN().to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    wandb.watch(model, log="all")

    # -----------------------------
    # 5. Training loop
    # -----------------------------
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_loss, val_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # -----------------------------
    # 6. Save model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    os.makedirs("deployment", exist_ok=True)
    torch.save(model.state_dict(), "deployment/mnist_cnn.pth")


if __name__ == "__main__":
    train()
