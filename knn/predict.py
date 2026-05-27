# K-Nearest Neighbours classifier.
# No training phase -- stores the training set and predicts by majority vote.
#
# ref: Cover & Hart, "Nearest neighbor pattern classification" (1967)
#      https://doi.org/10.1109/TIT.1967.1053964
#
# Usage:
#   python knn/predict.py --feature raw  --k 5
#   python knn/predict.py --feature hog  --k 5
#   python knn/predict.py --feature pca  --k 5  --pca_dims 50

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import config
from utils.dataPreprocessing import load_features, load_labels, load_pca, FEATURES_DIR
from utils.helperFunctions   import save_knn, print_results, plot_confusion_matrix

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def knn_predict(X_test, X_train, y_train, k):
    """
    Vectorised KNN prediction.
    Distances computed as: ||a-b||^2 = ||a||^2 + ||b||^2 - 2(a.b)
    Avoids explicit loops over the training set.
    """
    y_pred     = np.empty(len(X_test), dtype=y_train.dtype)
    batch_size = 500

    for start in range(0, len(X_test), batch_size):
        batch    = X_test[start : start + batch_size]
        sq_test  = np.sum(batch**2,   axis=1, keepdims=True)
        sq_train = np.sum(X_train**2, axis=1, keepdims=True)
        dists    = sq_test + sq_train.T - 2.0 * (batch @ X_train.T)
        dists    = np.sqrt(np.maximum(dists, 0.0))

        knn_idx    = np.argpartition(dists, k, axis=1)[:, :k]
        knn_labels = y_train[knn_idx]

        for i, labels in enumerate(knn_labels):
            counts            = np.bincount(labels, minlength=config.NUM_CLASSES)
            y_pred[start + i] = np.argmax(counts)

    return y_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",  default="raw", choices=["raw", "hog", "pca"])
    p.add_argument("--k",        type=int,       default=config.KNN_K)
    p.add_argument("--pca_dims", type=int,       default=config.PCA_DIMS)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.feature} features ...")
    X_train = load_features("train", args.feature if args.feature != "pca"
                            else f"pca")
    X_test  = load_features("test",  args.feature if args.feature != "pca"
                            else f"pca")
    y_train = load_labels("train")
    y_test  = load_labels("test")

    tag = f"{args.feature}_k{args.k}"
    if args.feature == "pca":
        tag = f"pca{args.pca_dims}_k{args.k}"

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    save_knn(X_train, y_train, os.path.join(MODELS_DIR, f"knn_{tag}.pkl"))

    print(f"Predicting (k={args.k}, feature={args.feature}, n_train={len(X_train)}) ...")
    y_pred = knn_predict(X_test, X_train, y_train, k=args.k)

    print_results(y_test, y_pred, model_name=f"KNN [{tag}]")
    plot_confusion_matrix(y_test, y_pred,
                          title=f"KNN [{tag}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{tag}.png"))


if __name__ == "__main__":
    main()