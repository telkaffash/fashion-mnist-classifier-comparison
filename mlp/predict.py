# Load a saved MLP and evaluate on the test set.
#
# Usage:
#   python mlp/predict.py --feature raw
#   python mlp/predict.py --feature pca --pca_dims 100

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import config
from utils.dataPreprocessing import load_dataset, get_features, load_pca
from utils.helperFunctions   import (load_mlp, print_results, plot_confusion_matrix)
from mlp.train               import forward_pass

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature",  default="raw", choices=["raw","enhanced","pca"])
    p.add_argument("--pca_dims", type=int,       default=config.PCA_DIMS)
    return p.parse_args()


def main():
    args = parse_args()
    tag  = f"pca{args.pca_dims}" if args.feature == "pca" else args.feature

    print("Loading data ...")
    _, _, _, _, X_test, y_test = load_dataset()

    pca = None
    if args.feature == "pca":
        pca = load_pca(os.path.join(MODELS_DIR, f"pca_{args.pca_dims}"))
    X_test_f, _ = get_features(X_test, args.feature, pca=pca)

    W, b     = load_mlp(os.path.join(MODELS_DIR, f"mlp_{tag}"))
    probs, _ = forward_pass(X_test_f, W, b)
    y_pred   = probs.argmax(axis=1)

    print_results(y_test, y_pred, model_name=f"MLP [{tag}]")
    plot_confusion_matrix(y_test, y_pred,
                          title=f"MLP [{tag}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{tag}_eval.png"))


if __name__ == "__main__":
    main()
