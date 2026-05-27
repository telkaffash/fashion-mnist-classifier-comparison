import os

DATA_DIR    = "data"
NUM_CLASSES = 10
IMG_SIZE    = 28
VAL_SPLIT   = 0.1
RANDOM_SEED = 42

CLASS_NAMES = [
    "T-shirt/Top", "Trouser",  "Pullover", "Dress",    "Coat",
    "Sandal",      "Shirt",    "Sneaker",  "Bag",      "Ankle boot"
]

# PCA
PCA_DIMS       = 50
PCA_SWEEP_DIMS = [10, 20, 50, 100, 200]

# KNN
KNN_K         = 5
KNN_SUBSAMPLE = 10000   # using a subset of the train set per batch for faster predictions

# Logistic Regression
LOGREG_LR         = 1e-3
LOGREG_EPOCHS     = 30
LOGREG_BATCH_SIZE = 256

# MLP
MLP_HIDDEN_SIZES = [512, 256]
MLP_LR           = 1e-3
MLP_EPOCHS       = 30
MLP_BATCH_SIZE   = 256
MLP_LAMBDA       = 1e-4
MLP_DROPOUT      = 0.3

# CNN
CNN_LR         = 1e-3
CNN_EPOCHS     = 60
CNN_BATCH_SIZE = 128

# SVM
SVM_LR         = 1e-3
SVM_EPOCHS     = 50
SVM_BATCH_SIZE = 256
SVM_LAMBDA     = 1e-2

# Adam
ADAM_BETA1   = 0.9
ADAM_BETA2   = 0.999
ADAM_EPSILON = 1e-8