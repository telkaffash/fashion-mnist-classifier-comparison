# Load a saved logistic regression model and evaluate on the test set.
#
# Usage:
#   python logistic_regression/predict.py --feature raw
#   python logistic_regression/predict.py --feature hog
#   python logistic_regression/predict.py --feature pca

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import config
from utils.dataPreprocessing         import load_features, load_labels
from utils.helperFunctions           import load_torch_model, print_results, plot_confusion_matrix
from logistic_regression.train       import build_model, predict

MODELS_DIR = os.path.join(os.path.dirname(__file__), "results", "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "results", "plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="raw", choices=["raw", "hog", "pca"])
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.feature} features ...")
    X_test = load_features("test", args.feature)
    y_test = load_labels("test")

    model  = load_torch_model(build_model(X_test.shape[1]),
                              os.path.join(MODELS_DIR, f"logreg_{args.feature}"))
    y_pred = predict(X_test, model)

    print_results(y_test, y_pred, model_name=f"Logistic Regression [{args.feature}]")
    plot_confusion_matrix(y_test, y_pred,
                          title=f"Logistic Regression [{args.feature}]",
                          save_path=os.path.join(PLOTS_DIR, f"cm_{args.feature}_eval.png"))


if __name__ == "__main__":
    main()