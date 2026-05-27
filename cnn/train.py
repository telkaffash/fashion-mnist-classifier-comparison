# Usage:
#   python cnn/train.py
#   python cnn/train.py --arch small
#   python cnn/train.py --arch medium
#   python cnn/train.py --arch deep

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from utils.dataPreprocessing import load_features, load_labels
from utils.helperFunctions   import (save_torch_model, print_results,
                                     plot_confusion_matrix, plot_training_curves,
                                     compute_accuracy)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


# === Architectures ===

class ConvNetSmall(nn.Module):
    # 1 conv block: Conv(32) -> BN -> ReLU -> Pool -> Dense(64) -> Out
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28x28 -> 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 32 * 14 * 14 = 6272
            nn.Linear(32 * 14 * 14, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ConvNetMedium(nn.Module):
    # 2 conv blocks: Conv(32) -> Pool -> Conv(64) -> Pool -> Dense(128) -> Out
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 64 * 7 * 7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ConvNetDeep(nn.Module):
    # 3 conv blocks: Conv(32) -> Pool -> Conv(64) -> Pool -> Conv(128) -> Dense(256) -> Out
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 14x14 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
                                        # no pool -- 7x7 is already small
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128 * 7 * 7 = 6272
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


ARCHITECTURES = {
    "small":  ConvNetSmall,
    "medium": ConvNetMedium,
    "deep":   ConvNetDeep,
}


# ---- Training -----------------------------------------------------------

def to_image_tensor(X_flat):
    return torch.tensor(X_flat.reshape(-1, 1, 28, 28), dtype=torch.float32)


def train(X_train, y_train, X_val, y_val, model, lr, epochs, batch_size, device,
          patience=5):
    """
    patience: stop if val loss does not improve for this many consecutive epochs.
    Best weights are restored at the end.
    ref: Prechelt, "Early Stopping -- But When?" (1998)
         https://doi.org/10.1007/3-540-49430-8_3
    """
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

    def augment(X_batch):
        mask = torch.rand(len(X_batch)) > 0.5
        X_batch[mask] = torch.flip(X_batch[mask], dims=[3])
        return X_batch

    train_dl  = DataLoader(
        TensorDataset(to_image_tensor(X_train),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)

    X_val_t = to_image_tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = augment(Xb).to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            idx     = np.random.choice(len(X_train), size=2000, replace=False)
            tr_sub  = to_image_tensor(X_train[idx]).to(device)
            tr_out  = model(tr_sub)
            tr_loss = loss_fn(tr_out, torch.tensor(y_train[idx], dtype=torch.long).to(device)).item()
            tr_acc  = compute_accuracy(y_train[idx], tr_out.argmax(1).cpu().numpy())

            vl_out  = model(X_val_t)
            vl_loss = loss_fn(vl_out, y_val_t).item()
            vl_acc  = compute_accuracy(y_val, vl_out.argmax(1).cpu().numpy())

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss     = vl_loss
            best_weights      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tag = "  *"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve {epochs_no_improve}/{patience})"

        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} -- restoring best weights.")
            break

    model.load_state_dict(best_weights)
    print(f"  Best val loss: {best_val_loss:.4f}")
    return train_losses, val_losses, train_accs, val_accs


def predict(X_flat, model, device, batch_size=256):
    model.eval()
    preds = []
    X_t   = to_image_tensor(X_flat)
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            out = model(X_t[start:start+batch_size].to(device))
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ---- Entry point --------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",       default="medium", choices=["small", "medium", "deep"])
    p.add_argument("--lr",         type=float,       default=config.CNN_LR)
    p.add_argument("--epochs",     type=int,         default=config.CNN_EPOCHS)
    p.add_argument("--batch_size", type=int,         default=config.CNN_BATCH_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Architecture: {args.arch}")

    print("Loading raw features ...")
    X_train = load_features("train", "raw")
    X_val   = load_features("val",   "raw")
    X_test  = load_features("test",  "raw")
    y_train = load_labels("train")
    y_val   = load_labels("val")
    y_test  = load_labels("test")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    model = ARCHITECTURES[args.arch]()
    print(f"Training CNN [{args.arch}]  lr={args.lr}  epochs={args.epochs} ...")
    tr_losses, vl_losses, tr_accs, vl_accs = train(
        X_train, y_train, X_val, y_val, model,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, device=device)

    save_torch_model(model, os.path.join(MODELS_DIR, f"cnn_{args.arch}"))

    y_pred = predict(X_test, model, device)
    print_results(y_test, y_pred, model_name=f"CNN [{args.arch}]")
    plot_training_curves(tr_losses, vl_losses, tr_accs, vl_accs,
                         title=f"CNN [{args.arch}]",
                         save_path=os.path.join(PLOTS_DIR, f"curves_cnn_{args.arch}.png"))
    plot_confusion_matrix(y_test, y_pred,
                          title=f"CNN [{args.arch}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_cnn_{args.arch}.png"))


if __name__ == "__main__":
    main()