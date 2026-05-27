# Usage:
#   python mlp/train.py --feature raw
#   python mlp/train.py --feature hog
#   python mlp/train.py --feature pca
#   python mlp/train.py --feature raw --hidden 512 256 128 --dropout 0.4

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import config
from utils.dataPreprocessing import load_features, load_labels
from utils.helperFunctions   import (adam_init, adam_step, save_mlp,
                                     print_results, plot_confusion_matrix,
                                     plot_training_curves, compute_accuracy)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def init_weights(layer_sizes, seed=config.RANDOM_SEED):
    # He initialization: std = sqrt(2 / fan_in), suitable for ReLU
    rng = np.random.default_rng(seed)
    W, b = [], []
    for i in range(len(layer_sizes) - 1):
        std = np.sqrt(2.0 / layer_sizes[i])
        W.append(rng.normal(0, std, (layer_sizes[i], layer_sizes[i+1])))
        b.append(np.zeros(layer_sizes[i+1]))
    return W, b


def cosine_lr(base_lr, epoch, total_epochs):
    # Cosine annealing: smoothly decay lr from base_lr to near 0
    return base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_with_smoothing(probs, y, smooth=0.0):
    """
    Cross-entropy loss with optional label smoothing.
    Smoothing replaces hard one-hot targets with:
        (1 - smooth) for the true class, smooth / (C-1) for others.
    This prevents overconfident predictions.
    """
    n, C    = probs.shape
    if smooth == 0.0:
        return -np.mean(np.log(probs[np.arange(n), y] + 1e-12))
    # Soft targets
    targets = np.full((n, C), smooth / (C - 1))
    targets[np.arange(n), y] = 1.0 - smooth
    return -np.mean(np.sum(targets * np.log(probs + 1e-12), axis=1))


def forward_pass(X, W, b, dropout_rate=0.0, training=False):
    rng   = np.random.default_rng()
    A     = X
    cache = []
    for i in range(len(W)):
        Z    = A @ W[i] + b[i]
        A_in = A
        if i < len(W) - 1:
            A    = np.maximum(0, Z)
            mask = None
            if training and dropout_rate > 0:
                mask = (rng.random(A.shape) > dropout_rate).astype(np.float64)
                A   *= mask
                A   /= (1.0 - dropout_rate)
        else:
            A    = softmax(Z)
            mask = None
        cache.append((Z, A_in, mask))
    return A, cache


def backward_pass(probs, y, W, cache, lam, dropout_rate, smooth=0.0):
    n, C = probs.shape
    dW   = [None] * len(W)
    db   = [None] * len(W)

    # Gradient of cross-entropy (with smoothing) w.r.t. softmax input
    if smooth == 0.0:
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y] = 1.0
    else:
        one_hot = np.full_like(probs, smooth / (C - 1))
        one_hot[np.arange(n), y] = 1.0 - smooth

    dA = (probs - one_hot) / n

    for i in reversed(range(len(W))):
        Z, A_in, mask = cache[i]
        if i < len(W) - 1:
            if mask is not None:
                dA /= (1.0 - dropout_rate)
                dA *= mask
            dZ = dA * (Z > 0).astype(np.float64)
        else:
            dZ = dA
        dW[i] = A_in.T @ dZ + lam * W[i]
        db[i] = dZ.sum(axis=0)
        dA    = dZ @ W[i].T
    return dW, db


def train(X_train, y_train, X_val, y_val, W, b,
          lr, epochs, batch_size, lam, dropout_rate, smooth, patience=5):
    """
    patience: stop if val loss does not improve for this many consecutive epochs.
    Best weights are restored at the end.
    ref: Prechelt, "Early Stopping -- But When?" (1998)
         https://doi.org/10.1007/3-540-49430-8_3
    """
    params = W + b
    rng    = np.random.default_rng(config.RANDOM_SEED)
    n      = len(X_train)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss     = float("inf")
    best_W            = None
    best_b            = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        current_lr = cosine_lr(lr, epoch, epochs)
        opt_state  = adam_init(params, lr=current_lr)

        idx    = rng.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train[idx]

        for start in range(0, n, batch_size):
            Xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            probs, cache = forward_pass(Xb, W, b, dropout_rate, training=True)
            dW, db       = backward_pass(probs, yb, W, cache, lam, dropout_rate, smooth)
            adam_step(params, dW + db, opt_state)

        tr_probs, _ = forward_pass(X_train, W, b)
        tr_loss     = cross_entropy_with_smoothing(tr_probs, y_train, smooth)
        tr_acc      = compute_accuracy(y_train, tr_probs.argmax(axis=1))

        vl_probs, _ = forward_pass(X_val, W, b)
        vl_loss     = cross_entropy_with_smoothing(vl_probs, y_val, smooth)
        vl_acc      = compute_accuracy(y_val, vl_probs.argmax(axis=1))

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss     = vl_loss
            best_W            = [w.copy() for w in W]
            best_b            = [bi.copy() for bi in b]
            epochs_no_improve = 0
            tag = "  *"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve {epochs_no_improve}/{patience})"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  lr={current_lr:.5f}"
                  f"  loss={tr_loss:.4f}  acc={tr_acc:.4f}"
                  f"  val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{tag}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} -- restoring best weights.")
            break

    # Restore best weights in-place
    for i in range(len(W)):
        W[i][:] = best_W[i]
        b[i][:] = best_b[i]

    print(f"  Best val loss: {best_val_loss:.4f}")
    return train_losses, val_losses, train_accs, val_accs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",    default="raw",  choices=["raw", "hog", "pca"])
    p.add_argument("--hidden",     type=int, nargs="+", default=config.MLP_HIDDEN_SIZES)
    p.add_argument("--lr",         type=float,     default=config.MLP_LR)
    p.add_argument("--epochs",     type=int,       default=config.MLP_EPOCHS)
    p.add_argument("--batch_size", type=int,       default=config.MLP_BATCH_SIZE)
    p.add_argument("--lam",        type=float,     default=config.MLP_LAMBDA)
    p.add_argument("--dropout",    type=float,     default=config.MLP_DROPOUT)
    p.add_argument("--smooth",     type=float,     default=0.1,
                   help="Label smoothing strength (0=disabled, 0.1=recommended)")
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

    layer_sizes = [X_train.shape[1]] + args.hidden + [config.NUM_CLASSES]
    W, b        = init_weights(layer_sizes)
    print(f"Architecture: {layer_sizes}  dropout={args.dropout}  smooth={args.smooth}")

    print(f"Training MLP [{args.feature}]  lr={args.lr}  epochs={args.epochs} ...")
    tr_losses, vl_losses, tr_accs, vl_accs = train(
        X_train, y_train, X_val, y_val, W, b,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        lam=args.lam, dropout_rate=args.dropout, smooth=args.smooth)

    save_mlp(W, b, os.path.join(MODELS_DIR, f"mlp_{args.feature}"))

    probs, _ = forward_pass(X_test, W, b)
    y_pred   = probs.argmax(axis=1)
    print_results(y_test, y_pred, model_name=f"MLP [{args.feature}]")
    plot_training_curves(tr_losses, vl_losses, tr_accs, vl_accs,
                         title=f"MLP [{args.feature}]",
                         save_path=os.path.join(PLOTS_DIR, f"curves_{args.feature}.png"))
    plot_confusion_matrix(y_test, y_pred,
                          title=f"MLP [{args.feature}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{args.feature}.png"))


if __name__ == "__main__":
    main()