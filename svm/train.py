# Usage:
#   python svm/train.py --feature raw
#   python svm/train.py --feature hog
#   python svm/train.py --feature pca
#   python svm/train.py --feature pca --lam 1e-2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import config
from utils.dataPreprocessing import load_features, load_labels
from utils.helperFunctions   import (adam_init, adam_step, save_svm,
                                     print_results, plot_confusion_matrix,
                                     compute_accuracy)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def train_one_vs_rest(X_train, y_train, lr, epochs, batch_size, lam):
    """
    Train one binary SVM per class.
    Returns list of (w, b) tuples, one per class.
    """
    n, d    = X_train.shape
    weights = []
    rng     = np.random.default_rng(config.RANDOM_SEED)

    for c in range(config.NUM_CLASSES):
        y_binary = np.where(y_train == c, 1.0, -1.0)
        w        = np.zeros(d)
        b        = 0.0
        opt      = adam_init([w], lr=lr)

        for epoch in range(epochs):
            idx    = rng.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_binary[idx]

            for start in range(0, n, batch_size):
                Xb       = X_shuf[start : start + batch_size]
                yb       = y_shuf[start : start + batch_size]
                margins  = yb * (Xb @ w + b)
                violated = margins < 1.0
                dw       = -(yb * violated)[:, None] * Xb
                dw       = dw.mean(axis=0) + lam * w
                db       = -(yb * violated).mean()
                adam_step([w], [dw], opt)
                b       -= lr * db

        weights.append((w, b))
        print(f"  Class {c+1:>2}/{config.NUM_CLASSES} done", end="\r")

    print()
    return weights


def predict_svm(X, weights):
    scores = np.column_stack([X @ w + b for w, b in weights])
    return scores.argmax(axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",    default="pca", choices=["raw", "hog", "pca"])
    p.add_argument("--lr",         type=float,    default=config.SVM_LR)
    p.add_argument("--epochs",     type=int,      default=config.SVM_EPOCHS)
    p.add_argument("--batch_size", type=int,      default=config.SVM_BATCH_SIZE)
    p.add_argument("--lam",        type=float,    default=config.SVM_LAMBDA)
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

    print(f"Training SVM (One-vs-Rest) [{args.feature}]  lr={args.lr}  epochs={args.epochs} ...")
    weights = train_one_vs_rest(X_train, y_train,
                                lr=args.lr, epochs=args.epochs,
                                batch_size=args.batch_size, lam=args.lam)

    val_pred = predict_svm(X_val, weights)
    print(f"Val accuracy: {compute_accuracy(y_val, val_pred):.4f}")

    save_svm(weights, os.path.join(MODELS_DIR, f"svm_{args.feature}.pkl"))

    y_pred = predict_svm(X_test, weights)
    print_results(y_test, y_pred, model_name=f"SVM [{args.feature}]")
    plot_confusion_matrix(y_test, y_pred,
                          title=f"SVM [{args.feature}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{args.feature}.png"))


if __name__ == "__main__":
    main()