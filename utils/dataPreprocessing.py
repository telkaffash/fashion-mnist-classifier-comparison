'''
=========================== dataPreprocessing.py ====================================

This code loads and splits the dataset, and applies feature engineering techniques.

>> Notes
    - There are three feature modes: raw, hog, pca.
    - Always run this script first before any classifier.
    - Running may take some time.

>> Usage:
    $ python utils/dataPreprocessing.py
    $ python utils/dataPreprocessing.py --show --pca_dims 180

'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")) # I borrowed this shortcut from: https://gist.github.com/justinmklam/c453ea62b47f4e302dbd83c05882e3bd
import argparse
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import config

FMNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# Preprocessed features are saved in this directory to be loaded by all classifiers
FEATURES_DIR = "features/"

def load_images_from_file(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(n, rows, cols).astype(np.float32)   # (N, 28, 28)

def load_labels_from_file(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


# === Load and split the dataset ===
def load_dataset(data_dir=config.DATA_DIR, val_split=config.VAL_SPLIT,
                 subsample=None, seed=config.RANDOM_SEED):
    # Returns images in (N, 28, 28) float32 normalized to [0, 1].
    X_all  = load_images_from_file(os.path.join(data_dir, FMNIST_FILES["train_images"])) / 255.0
    y_all  = load_labels_from_file(os.path.join(data_dir, FMNIST_FILES["train_labels"]))
    X_test = load_images_from_file(os.path.join(data_dir, FMNIST_FILES["test_images"]))  / 255.0
    y_test = load_labels_from_file(os.path.join(data_dir, FMNIST_FILES["test_labels"]))

    if subsample is not None and subsample < len(X_all): ### FIX (remove) ++ also remove eigenfaces plotREMO
        rng   = np.random.default_rng(seed)
        idx   = rng.choice(len(X_all), size=subsample, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    # Splits training data into train/val
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(len(X_all))
    n_val   = int(len(X_all) * val_split)
    X_train = X_all[idx[n_val:]]
    y_train = y_all[idx[n_val:]]
    X_val   = X_all[idx[:n_val]]
    y_val   = y_all[idx[:n_val]]

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# === Save processed features ===
def save_features(split, feature_name, data):
    # split = "train", "val", or "test"
    # feature_name = "raw", "hog", or "pca"
    os.makedirs(FEATURES_DIR, exist_ok=True)
    path = os.path.join(FEATURES_DIR, f"{split}_{feature_name}.npy")
    np.save(path, data)

# === Load for reuse ===
def load_features(split, feature_name):
    path = os.path.join(FEATURES_DIR, f"{split}_{feature_name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Features not found at {path}. Run dataPreprocessing.py first.")
    return np.load(path)

def load_labels(split):
    path = os.path.join(FEATURES_DIR, f"{split}_labels.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labels not found at {path}. Run dataPreprocessing.py first.")
    return np.load(path)

# === Feature engineering ===
def flatten(X):
    # (N, 28, 28) -> (N, 784)
    return X.reshape(len(X), -1)

def apply_hog(X, cell_size=4, n_bins=9):
    """
    Histogram of Oriented Gradients (HOG) from scratch.
    Captures edge directions and shapes rather than raw pixel intensities.
    ref: Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection" (2005)
         https://doi.org/10.1109/CVPR.2005.177

    Steps per image:
        1. Compute x and y gradients using [-1, 0, 1] filters
        2. Compute gradient magnitude and orientation at each pixel
        3. Divide image into cells (cell_size x cell_size pixels)
        4. Build a weighted histogram of orientations per cell
        5. L2-normalize and concatenate all cell histograms
    Output shape: (N, n_cells_h * n_cells_w * n_bins) = (N, 441) for 28x28, cell=4, bins=9
    """
    from numpy.lib.stride_tricks import sliding_window_view

    N, H, W    = X.shape[0], X.shape[1], X.shape[2]
    n_cells_h  = H // cell_size
    n_cells_w  = W // cell_size
    out        = np.zeros((N, n_cells_h * n_cells_w * n_bins), dtype=np.float32)
    bin_edges  = np.linspace(0, np.pi, n_bins + 1)
    kx         = np.array([[-1, 0, 1]], dtype=np.float32)
    ky         = kx.T

    for i in range(N):
        img = X[i]

        px          = np.pad(img, ((0,0),(1,1)), mode="reflect")
        py          = np.pad(img, ((1,1),(0,0)), mode="reflect")
        gx          = (sliding_window_view(px, (1,3)).squeeze(2) * kx[0]).sum(-1)
        gy          = (sliding_window_view(py, (3,1)).squeeze(3) * ky[:,0]).sum(-1)
        magnitude   = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(np.abs(gy), np.abs(gx))   # unsigned, range [0, pi]

        hist = np.zeros((n_cells_h, n_cells_w, n_bins), dtype=np.float32)
        for ch in range(n_cells_h):
            for cw in range(n_cells_w):
                r0, r1 = ch * cell_size, (ch+1) * cell_size
                c0, c1 = cw * cell_size, (cw+1) * cell_size
                hist[ch, cw] = np.histogram(
                    orientation[r0:r1, c0:c1].ravel(), bins=bin_edges,
                    weights=magnitude[r0:r1, c0:c1].ravel())[0]

        flat    = hist.ravel()
        out[i]  = flat / (np.linalg.norm(flat) + 1e-6)

        if i % 1000 == 0:
            print(f"    {i}/{N} images processed...", end="\r")

    print()
    return out

def fit_pca(X_flat, n_components):
    """
    PCA via eigendecomposition of the covariance matrix.
    ref: Bishop, "Pattern Recognition and Machine Learning", Ch. 12
         https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/

    X_flat: (N, 784)
    Returns a pca dict with mean, components, explained variance.
    """
    mean      = X_flat.mean(axis=0)
    X_centred = X_flat - mean
    cov       = (X_centred.T @ X_centred) / (len(X_flat) - 1)

    # eigh is optimized for symmetric matrices, returns ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx          = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components    = eigenvectors[:, :n_components]
    explained_var = eigenvalues[:n_components]
    total_var     = eigenvalues.sum()

    print(f"PCA: {n_components} components capture "
          f"{explained_var.sum() / total_var * 100:.1f}% of variance")

    return dict(mean=mean, components=components,
                explained_var=explained_var, total_var=total_var,
                n_components=n_components)

def apply_pca(X_flat, pca):
    return (X_flat - pca["mean"]) @ pca["components"]   # (N, n_components)

def inverse_pca(Z, pca):
    return Z @ pca["components"].T + pca["mean"]

def save_pca(pca, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=pca["mean"], components=pca["components"],
             explained_var=pca["explained_var"],
             total_var=[pca["total_var"]], n_components=[pca["n_components"]])

def load_pca(path):
    d = np.load(path + ".npz")
    return dict(mean=d["mean"], components=d["components"],
                explained_var=d["explained_var"],
                total_var=float(d["total_var"][0]),
                n_components=int(d["n_components"][0]))

def get_features(X_images, feature, pca=None, fit=False, n_components=config.PCA_DIMS):
    if feature == "raw":
        return flatten(X_images), None

    if feature == "hog":
        return apply_hog(X_images), None

    if feature == "pca":
        X_flat = flatten(X_images)
        if fit:
            pca = fit_pca(X_flat, n_components)
        return apply_pca(X_flat, pca), pca

    raise ValueError(f"Unknown feature type '{feature}'. Choose: raw, hog, pca")

# === Compute and save all features ===
def run_preprocessing(pca_dims=config.PCA_DIMS):
    """
    Load dataset, compute all three feature sets, and save to features/.
    Run this once before training any model.
    """
    print("\n[1/4] Loading dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Save labels
    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.save(os.path.join(FEATURES_DIR, "train_labels.npy"), y_train)
    np.save(os.path.join(FEATURES_DIR, "val_labels.npy"),   y_val)
    np.save(os.path.join(FEATURES_DIR, "test_labels.npy"),  y_test)
    print("Labels saved.")

    # Raw
    print("\n[2/4] Computing raw features...")
    for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        save_features(split, "raw", flatten(X))
    print("Raw features saved.")

   # HOG
    print("\n[3/4] Computing HOG features...")
    for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        print(f"  HOG {split} set ({len(X)} images)...")
        save_features(split, "hog", apply_hog(X))
    print("HOG features saved.")

    # PCA
    print(f"\n[4/4] Computing PCA features (n_components={pca_dims})...")
    X_train_flat = flatten(X_train)
    pca          = fit_pca(X_train_flat, pca_dims)
    save_pca(pca, os.path.join(FEATURES_DIR, f"pca_{pca_dims}"))
    for split, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        save_features(split, "pca", apply_pca(flatten(X), pca))
    print("PCA features saved.")

    print(f"\nAll features saved to {FEATURES_DIR}")
    print("Classifiers can now load features with load_features(split, feature_name).")
    return X_train, y_train, pca

# === Visualization ===
def show_visualizations(X_train, y_train, pca, save_dir="results/pca_plots/"):
    os.makedirs(save_dir, exist_ok=True)

    target_classes = [0, 9]
    idx      = [np.where(y_train == c)[0][0] for c in target_classes]
    samples  = X_train[idx]
    labels   = [config.CLASS_NAMES[y_train[i]] for i in idx]
    pca_recon= np.clip(inverse_pca(apply_pca(flatten(samples), pca), pca).reshape(-1,28,28), 0, 1)

    # Compute HOG internals for visualization (gradient lines, not the flat vector)
    from numpy.lib.stride_tricks import sliding_window_view
    cell_size = 4
    n_bins    = 9
    kx        = np.array([[-1, 0, 1]], dtype=np.float32)
    ky        = kx.T
    bin_edges = np.linspace(0, np.pi, n_bins + 1)

    def get_hog_cells(img):
        px  = np.pad(img, ((0,0),(1,1)), mode="reflect")
        py  = np.pad(img, ((1,1),(0,0)), mode="reflect")
        gx  = (sliding_window_view(px, (1,3)).squeeze(2) * kx[0]).sum(-1)
        gy  = (sliding_window_view(py, (3,1)).squeeze(3) * ky[:,0]).sum(-1)
        mag = np.sqrt(gx**2 + gy**2)
        ori = np.arctan2(np.abs(gy), np.abs(gx))
        n_cells_h = img.shape[0] // cell_size
        n_cells_w = img.shape[1] // cell_size
        hist = np.zeros((n_cells_h, n_cells_w, n_bins))
        for ch in range(n_cells_h):
            for cw in range(n_cells_w):
                r0,r1 = ch*cell_size, (ch+1)*cell_size
                c0,c1 = cw*cell_size, (cw+1)*cell_size
                hist[ch,cw] = np.histogram(ori[r0:r1,c0:c1].ravel(),
                                           bins=bin_edges,
                                           weights=mag[r0:r1,c0:c1].ravel())[0]
        return hist

    def draw_hog_lines(ax, img, hist):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.40)
        bin_centers   = np.linspace(0, np.pi, n_bins, endpoint=False)
        n_cells_h, n_cells_w, _ = hist.shape
        for ch in range(n_cells_h):
            for cw in range(n_cells_w):
                cx       = cw * cell_size + cell_size / 2
                cy       = ch * cell_size + cell_size / 2
                dominant = bin_centers[np.argmax(hist[ch, cw])]
                strength = hist[ch, cw].max() / (hist.max() + 1e-6) * cell_size * 0.9
                dx = np.cos(dominant) * strength / 2
                dy = np.sin(dominant) * strength / 2
                ax.plot([cx-dx, cx+dx], [cy-dy, cy+dy], color="lime", linewidth=1.2)
        ax.axis("off")

    # Figure 1: Raw | HOG overlay | PCA reconstruction
    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    for col, title in enumerate(["Raw", "HOG", f"PCA ({pca['n_components']} dims)"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, (raw, rec, label) in enumerate(zip(samples, pca_recon, labels)):
        axes[row, 0].imshow(raw, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        draw_hog_lines(axes[row, 1], raw, get_hog_cells(raw))
        axes[row, 2].imshow(rec, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=50, va="center")

    fig.suptitle("Feature Comparison: Raw vs HOG vs PCA", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "feature_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # Figure 2: Cumulative explained variance curve
    cumvar = np.cumsum(pca["explained_var"]) / pca["total_var"] * 100
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(range(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=2)
    ax3.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.15, color="steelblue")
    colors = plt.cm.tab10(np.linspace(0, 1, len(config.PCA_SWEEP_DIMS)))
    for dim, color in zip(config.PCA_SWEEP_DIMS, colors):
        if dim <= len(cumvar):
            ax3.axvline(x=dim, color=color, linestyle="--", linewidth=1.2)
            ax3.annotate(f"{dim}D  {cumvar[dim-1]:.0f}%",
                         xy=(dim, cumvar[dim-1]), xytext=(dim+3, cumvar[dim-1]-7),
                         fontsize=8, color=color,
                         arrowprops=dict(arrowstyle="->", color=color, lw=1))
    ax3.axhline(y=95, color="gray", linestyle=":", linewidth=1)
    ax3.set_xlabel("Number of Components")
    ax3.set_ylabel("Cumulative Explained Variance (%)")
    ax3.set_title("PCA Explained Variance")
    ax3.set_ylim(0, 101)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    path = os.path.join(save_dir, "explained_variance.png")
    fig3.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    # Figure 3: PC1 vs PC2 scatter colored by class :)
    n_scatter   = 3000
    rng_idx     = np.random.default_rng(config.RANDOM_SEED).choice(
                      len(X_train), size=n_scatter, replace=False)
    X_projected = apply_pca(flatten(X_train[rng_idx]), pca)
    y_subset    = y_train[rng_idx]
    colors_cls  = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))

    fig4, ax4 = plt.subplots(figsize=(10, 7))
    for c in range(config.NUM_CLASSES):
        mask = y_subset == c
        ax4.scatter(X_projected[mask, 0], X_projected[mask, 1],
                    color=colors_cls[c], label=config.CLASS_NAMES[c],
                    alpha=0.4, s=10, linewidths=0)
    scale = X_projected[:, 0].std() * 2
    ax4.annotate("", xy=(scale, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="navy", lw=2))
    ax4.annotate("", xy=(0, scale * 0.6), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="navy", lw=2))
    ax4.text(scale * 1.05, 0,            "PC1", color="navy", fontsize=11, fontweight="bold")
    ax4.text(0,            scale * 0.65, "PC2", color="navy", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Principal Component 1")
    ax4.set_ylabel("Principal Component 2")
    ax4.set_title("Fashion-MNIST projected onto PC1 and PC2")
    ax4.legend(markerscale=2, fontsize=8, loc="upper left", framealpha=0.7, ncol=2)
    ax4.grid(alpha=0.2)
    fig4.tight_layout()
    path = os.path.join(save_dir, "pca_scatter.png")
    fig4.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Fashion-MNIST and save all feature sets.")
    parser.add_argument("--show",     action="store_true",
                        help="Show and save visualization figures after preprocessing.")
    parser.add_argument("--pca_dims", type=int, default=config.PCA_DIMS,
                        help=f"Number of PCA components (default: {config.PCA_DIMS}).")
    args = parser.parse_args()

    X_train, y_train, pca = run_preprocessing(pca_dims=args.pca_dims)

    if args.show:
        print("\nGenerating visualizations...")
        show_visualizations(X_train, y_train, pca)