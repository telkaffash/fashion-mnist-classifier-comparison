import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def compute_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def compute_per_class_f1(y_true, y_pred):
    f1_scores = []
    for c in range(config.NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)
    return f1_scores

def compute_macro_f1(y_true, y_pred):
    return float(np.mean(compute_per_class_f1(y_true, y_pred)))

def print_results(y_true, y_pred, model_name):
    n          = config.NUM_CLASSES
    overall    = compute_accuracy(y_true, y_pred)
    f1_scores  = compute_per_class_f1(y_true, y_pred)
    macro_f1   = float(np.mean(f1_scores))

    # TP, FP, and FN for precision and recall
    print(f"\n{'-'*65}")
    print(f"  {model_name}")
    print(f"{'-'*65}")
    print(f"  {'Class':<14}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'':>2}")
    print(f"  {'-'*61}")

    for c, name in enumerate(config.CLASS_NAMES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = f1_scores[c]
        bar       = "█" * int(f1 * 20) # For that cool loading effect

        print(f"  {name:<14}  {precision:>9.3f}  {recall:>7.3f}  {f1:>6.3f}  {bar}")

    print(f"  {'-'*61}")
    print(f"  {'Macro avg':<14}  {'':>9}  {'':>7}  {macro_f1:>6.3f}")
    print(f"  Overall accuracy: {overall:.4f}  ({overall*100:.2f}%)")
    print(f"{'-'*65}\n")
    return overall, macro_f1

def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    n  = config.NUM_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
            
    fig.tight_layout()
    save_figure(fig, save_path)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss"); ax1.legend()

    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs,   label="Val", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} - Accuracy"); ax2.legend()

    fig.tight_layout()
    save_figure(fig, save_path)

def adam_init(params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    return dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, t=0,
                m=[np.zeros_like(p) for p in params],
                v=[np.zeros_like(p) for p in params])

def adam_step(params, grads, state):
    state["t"] += 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        state["m"][i] = state["beta1"] * state["m"][i] + (1 - state["beta1"]) * g
        state["v"][i] = state["beta2"] * state["v"][i] + (1 - state["beta2"]) * g**2
        m_hat = state["m"][i] / (1 - state["beta1"]**t)
        v_hat = state["v"][i] / (1 - state["beta2"]**t)
        p    -= state["lr"] * m_hat / (np.sqrt(v_hat) + state["epsilon"])

def save_knn(X_train, y_train, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"X_train": X_train, "y_train": y_train}, f)
    print(f"KNN saved: {path}")

def load_knn(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"KNN loaded: {path}")
    return data["X_train"], data["y_train"]

def save_mlp(W, b, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {f"W{i}": w for i, w in enumerate(W)}
    arrays.update({f"b{i}": bi for i, bi in enumerate(b)})
    np.savez(path, **arrays)
    print(f"MLP saved: {path}.npz")

def load_mlp(path):
    data     = np.load(path + ".npz")
    n_layers = sum(1 for k in data if k.startswith("W"))
    W = [data[f"W{i}"] for i in range(n_layers)]
    b = [data[f"b{i}"] for i in range(n_layers)]
    print(f"MLP loaded: {path}")
    return W, b

def save_torch_model(model, path):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path + ".pt")
    print(f"Model saved: {path}.pt")

def load_torch_model(model, path):
    import torch
    model.load_state_dict(torch.load(path + ".pt", weights_only=True))
    model.eval()
    print(f"Model loaded: {path}.pt")
    return model


def save_svm(weights, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(weights, f)
    print(f"SVM saved: {path}")


def load_svm(path):
    with open(path, "rb") as f:
        weights = pickle.load(f)
    print(f"SVM loaded: {path}")
    return weights