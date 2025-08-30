from dataclasses import dataclass
import os
import pandas as pd
from typing import Tuple, Dict, List
from dataset import PlantDataset
from model import PlantNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging


@dataclass
class Config:
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 50
    lr: float = 0.001
    train_img_dir: str = "./datasets/plant-seedlings-classification/train"
    test_img_dir: str = "./datasets/plant-seedlings-classification/test"
    model_save_path: str = "./results/best.pth"
    loss_curve_save_path: str = "./results/loss_curve.png"
    confusion_matrix_save_path: str = "./results/confusion_matrix.png"
    submission_save_path: str = "./results/submission.csv"
    patience: int = 10


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


def get_species() -> Dict[str, int]:
    return {
        "Black-grass": 0,
        "Charlock": 1,
        "Cleavers": 2,
        "Common Chickweed": 3,
        "Common wheat": 4,
        "Fat Hen": 5,
        "Loose Silky-bent": 6,
        "Maize": 7,
        "Scentless Mayweed": 8,
        "Shepherds Purse": 9,
        "Small-flowered Cranesbill": 10,
        "Sugar beet": 11,
    }


def get_specie_names() -> List[str]:
    return list(get_species().keys())


def get_train_data(config: Config) -> List[Dict[str, str]]:
    train_data = []
    for specie_name in get_specie_names():
        species_dir = os.path.join(config.train_img_dir, specie_name)
        if os.path.exists(species_dir):
            for img_file in os.listdir(species_dir):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    train_data.append(
                        {
                            "image": os.path.join(specie_name, img_file),
                            "species": specie_name,
                        }
                    )
    if not train_data:
        raise ValueError("No training images found")
    return train_data


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Training)"
    )
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        train_loader_tqdm.set_postfix(loss=loss.item())
    return running_loss / len(train_loader.dataset)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    epochs: int,
) -> Tuple[float, List[int], List[int]]:
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        val_loader_tqdm = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{epochs} (Validation)"
        )
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0)
            val_loader_tqdm.set_postfix(loss=loss.item())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = running_val_loss / len(val_loader.dataset)
    return avg_val_loss, all_preds, all_labels


def ensure_dir_exists(file_path: str):
    dir = os.path.dirname(file_path)
    if dir:
        os.makedirs(dir, exist_ok=True)


def plot_loss_curve(config: Config, train_losses: List[float], val_losses: List[float]):
    save_path = config.loss_curve_save_path
    ensure_dir_exists(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss curve saved to '{save_path}'")


def plot_confusion_matrix(
    config: Config, labels: List[int], preds: List[int], species_names: List[str]
):
    save_path = config.confusion_matrix_save_path
    ensure_dir_exists(save_path)
    cm = confusion_matrix(labels, preds)
    _, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=species_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
    plt.title("Confusion Matrix on Validation Set")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Confusion matrix saved to '{save_path}'.")


def create_submission(
    config: Config,
    model: nn.Module,
    device: str,
    species_to_idx: Dict[str, int],
):
    model.eval()
    test_data = []
    if not os.path.exists(config.test_img_dir):
        raise ValueError("Test image directory does not exist")
    for img_file in os.listdir(config.test_img_dir):
        if img_file.endswith((".png", ".jpg", ".jpeg")):
            test_data.append({"image": img_file})
    test_df = pd.DataFrame(test_data)
    test_dataset = PlantDataset(
        dataframe=test_df,
        img_dir=config.test_img_dir,
        img_size=config.img_size,
        species_to_idx=species_to_idx,
        augment=False,
        is_test=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    idx_to_species = {v: k for k, v in species_to_idx.items()}
    predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Generating submission file"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs, 1)
            for idx in predicted_indices:
                predicted_species = idx_to_species[idx.item()]
                predictions.append(predicted_species)
    test_df["species"] = predictions
    submission_path = config.submission_save_path
    ensure_dir_exists(submission_path)
    submission_df = test_df.rename(columns={"image": "file"})
    submission_df.to_csv(submission_path, index=False)
    logging.info(f"submission file created at '{submission_path}'")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    config = Config()
    device = get_device()
    logging.info(f"Using device: {device}")
    species_to_idx = get_species()
    species_names = get_specie_names()
    num_classes = len(species_to_idx)
    logging.info(f"Number of classes: {num_classes}")
    train_data = get_train_data(config)
    full_df = pd.DataFrame(train_data)
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["species"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_dataset = PlantDataset(
        dataframe=train_df,
        img_dir=config.train_img_dir,
        img_size=config.img_size,
        species_to_idx=species_to_idx,
        augment=True,
    )
    val_dataset = PlantDataset(
        dataframe=val_df,
        img_dir=config.train_img_dir,
        img_size=config.img_size,
        species_to_idx=species_to_idx,
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    model = PlantNet(num_classes=num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.resnet.layer4.parameters(), "lr": config.lr * 0.1},
            {"params": model.resnet.fc.parameters(), "lr": config.lr},
        ]
    )
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    logging.info("Starting training")
    for epoch in range(config.epochs):
        epoch_train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=config.epochs,
        )
        train_losses.append(epoch_train_loss)
        epoch_val_loss, all_preds, all_labels = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=config.epochs,
        )
        val_losses.append(epoch_val_loss)
        logging.info(
            f"Epoch {epoch+1}/{config.epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}"
        )
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            logging.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logging.info(
                f"Validation loss did not improve, patience: {patience_counter}/{config.patience}"
            )
        if patience_counter >= config.patience:
            logging.info("Early stopping triggered.")
            break
    plot_loss_curve(config=config, train_losses=train_losses, val_losses=val_losses)
    plot_confusion_matrix(
        config=config,
        labels=all_labels,
        preds=all_preds,
        species_names=species_names,
    )
    create_submission(
        config=config, model=model, device=device, species_to_idx=species_to_idx
    )
    logging.info("Training completed")


if __name__ == "__main__":
    main()
