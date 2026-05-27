# Fashion-MNIST Classifier Comparison

This project evaluates multiple machine learning classifiers on the Fashion-MNIST dataset using different feature engineering techniques, with the goal of understanding how classifier choice and feature representation can affect classification performance.

The project compares:

- Raw pixel features
- PCA-reduced features
- HOG features

On the following classifiers:

- k-Nearest Neighbors
- Support Vector Machine
- Logistic Regression
- Multilayer Perceptron
- Convolutional Neural Network

The best perfoming model was a **deep (3 conv. layers, 32-64-128 maps) CNN model**, achieving **93.01% accuracy** and **93.0% macro-F1 score** on the Fashion-MNIST test set.

---

# How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing

```bash
python utils/dataPreprocessing.py --show
```

Or, to specify the PCA dimension:

```bash
python utils/dataPreprocessing.py --show --pca_dims 100
```

The default PCA dimension is 50 and can be adjusted in `config.py`.

### 3. Run classifiers

At the beginning of each script, you will find the usage cases in the first comment.

Make sure you always run `train.py` before `predict.py`, except for k-NN.

Examples:

```bash
python knn/predict.py --feature hog --k 5

python logistic_regression/train.py --feature raw
python logistic_regression/predict.py --feature raw

python cnn/train.py --arch deep
python cnn/predict.py --arch deep

```
# Notes

This project was initially completed as part of the Optimization and Machine Learning (ELEC449/MECE465) major elective course at the College of Engineering, Qatar University, and was later refined and expanded.

A full report detailing experimental results is included in: `Report.pdf`