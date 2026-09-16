"""Train a Food-11 classifier (ResNet18) and track the run with MLflow.

Loads one of the processed Food-11 datasets (data/food11_processed or
data/food11_processed_mini, see --dataset) with torchvision's ImageFolder,
fine-tunes a pretrained ResNet18 (final layer swapped for 11 classes), and
logs hyperparameters/metrics/the trained model to a local MLflow tracking
server for every run.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32

Requires a tracking server running at http://127.0.0.1:5000, e.g.:
    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
"""

import argparse
import time

import mlflow
import mlflow.pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_ROOT = "data"
NUM_CLASSES = 11

# Standard ImageNet normalization stats, since we're starting from ImageNet-pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
        help="Which processed dataset to train on (default: mini, for fast iteration)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_dataloaders(dataset_folder: str, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(f"{dataset_folder}/training", transform=transform)
    val_ds = datasets.ImageFolder(f"{dataset_folder}/validation", transform=transform)
    test_ds = datasets.ImageFolder(f"{dataset_folder}/evaluation", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.enable_grad() if train else torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    dataset_folder = (
        f"{DATA_ROOT}/food11_processed_mini" if args.dataset == "mini" else f"{DATA_ROOT}/food11_processed"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = build_dataloaders(dataset_folder, args.batch_size)
    print(f"Loaded {len(classes)} classes from {dataset_folder}: {classes}")

    # A single sample batch mlflow can use to trace/record the model's expected input shape.
    input_example = next(iter(train_loader))[0][:1].numpy()

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
            }
        )

        for epoch in range(args.epochs):
            start = time.time()
            train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            elapsed = time.time() - start

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch + 1}/{args.epochs} - train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} ({elapsed:.1f}s)"
            )

        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"final test_accuracy={test_accuracy:.4f}")

        mlflow.pytorch.log_model(model, "model", input_example=input_example)

    print("Done.")


if __name__ == "__main__":
    main()
