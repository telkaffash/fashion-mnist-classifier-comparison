# Usage:
#   python logistic_regression/train.py --feature raw
#   python logistic_regression/train.py --feature hog
#   python logistic_regression/train.py --feature pca
#   python logistic_regression/train.py --feature pca --epochs 50

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import config
from utils.dataPreprocessing import load_features, load_labels
from utils.helperFunctions   import (save_torch_model, print_results,
                                     plot_confusion_matrix, plot_training_curves,
                                     compute_accuracy)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def build_model(input_size):
    # Single linear layer -- this is logistic regression
    return nn.Linear(input_size, config.NUM_CLASSES)


def train(X_train, y_train, X_val, y_val, model, lr, epochs, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    train_dl  = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            loss_fn(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tr_out  = model(torch.tensor(X_train, dtype=torch.float32))
            tr_loss = loss_fn(tr_out, torch.tensor(y_train, dtype=torch.long)).item()
            tr_acc  = compute_accuracy(y_train, tr_out.argmax(1).numpy())

            vl_out  = model(X_val_t)
            vl_loss = loss_fn(vl_out, y_val_t).item()
            vl_acc  = compute_accuracy(y_val, vl_out.argmax(1).numpy())

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  "
                  f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs


def predict(X, model):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    return out.argmax(dim=1).numpy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",    default="raw", choices=["raw", "hog", "pca"])
    p.add_argument("--lr",         type=float,    default=config.LOGREG_LR)
    p.add_argument("--epochs",     type=int,      default=config.LOGREG_EPOCHS)
    p.add_argument("--batch_size", type=int,      default=config.LOGREG_BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.feature} features ...")
    X_train = load_features("train", args.feature)
    X_val   = load_features("val",   args.feature)
    X_test  = load_features("test",  args.feature)
    y_train = load_labels("train")
    y_val   = load_labels("val")
    y_test  = load_labels("test")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    model = build_model(X_train.shape[1])

    print(f"Training logistic regression [{args.feature}]  lr={args.lr}  epochs={args.epochs} ...")
    tr_losses, vl_losses, tr_accs, vl_accs = train(
        X_train, y_train, X_val, y_val, model,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)

    save_torch_model(model, os.path.join(MODELS_DIR, f"logreg_{args.feature}"))

    y_pred = predict(X_test, model)
    print_results(y_test, y_pred, model_name=f"Logistic Regression [{args.feature}]")
    plot_training_curves(tr_losses, vl_losses, tr_accs, vl_accs,
                         title=f"LogReg [{args.feature}]",
                         save_path=os.path.join(PLOTS_DIR, f"curves_{args.feature}.png"))
    plot_confusion_matrix(y_test, y_pred,
                          title=f"Logistic Regression [{args.feature}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{args.feature}.png"))


if __name__ == "__main__":
    main()