# Load a saved CNN and evaluate on the test set.
#
# Usage:
#   python cnn/predict.py --arch small
#   python cnn/predict.py --arch medium
#   python cnn/predict.py --arch deep

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import config
from utils.dataPreprocessing import load_features, load_labels
from utils.helperFunctions   import load_torch_model, print_results, plot_confusion_matrix
from cnn.train               import ARCHITECTURES, predict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="medium", choices=["small", "medium", "deep"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading raw features ...")
    X_test = load_features("test", "raw")
    y_test = load_labels("test")

    model  = load_torch_model(ARCHITECTURES[args.arch](),
                              os.path.join(MODELS_DIR, f"cnn_{args.arch}"))
    y_pred = predict(X_test, model, device)

    print_results(y_test, y_pred, model_name=f"CNN [{args.arch}]")
    plot_confusion_matrix(y_test, y_pred,
                          title=f"CNN [{args.arch}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_cnn_{args.arch}_eval.png"))


if __name__ == "__main__":
    main()