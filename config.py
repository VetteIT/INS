"""
Konfigurácia projektu - všetky hyperparametre na jednom mieste.
Vzor: Cvičenie 4 - FF.ipynb (hyperparametre ako premenné na začiatku kódu)

Semestrálny projekt: Porovnanie domain adaptation techník
pre detekciu Parkinsonovej choroby z reči
"""

import torch
import os

# ---- Zariadenie (GPU/CPU) ----
# Vzor z Cvičenie 3 - Komponenty neurónových sietí.ipynb
# https://pytorch.org/docs/stable/cuda.html
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Cesty ----
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ---- Hyperparametre trénovania ----
# Vzor z Cvičenie 4 - FF.ipynb: batch_size, learning_rate, num_epochs
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DA_EPOCHS = 100        # viac epoch pre domain adaptation (adversariálne trénovanie)
DA_LR = 0.0005         # nižší learning rate pre stabilnejšie DA
RANDOM_SEED = 42

# ---- Veľkosti vrstiev ----
# Vzor z Cvičenie 3: parameterizovaný návrh sietí
HIDDEN_SIZE = 64       # skrytá vrstva
FEATURE_DIM = 32       # dimenzia extrahovaných príznakov

# ---- Regularizácia ----
# Dropout a gradient clipping - prevencia overfittingu pre malé datasety
# https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
DROPOUT_RATE = 0.3
GRAD_CLIP_NORM = 1.0   # max norma gradientov
WEIGHT_DECAY = 1e-4    # L2 regularizácia v optimalizátore

# ---- Domain adaptácia ----
# DANN: Ganin, Y. & Lempitsky, V. (2015)
# "Unsupervised Domain Adaptation by Backpropagation"
# https://arxiv.org/abs/1409.7495
DANN_LAMBDA = 0.5      # maximálna sila adversariálneho signálu

# MMD: Gretton, A. et al. (2012)
# "A Kernel Two-Sample Test"
# https://jmlr.org/papers/v13/gretton12a.html
MMD_LAMBDA = 0.3

# ---- Dáta ----
# 12 spoločných akustických príznakov z oboch datasetov
NUM_FEATURES = 12
NUM_CLASSES = 2    # 0 = Healthy, 1 = Parkinson's Disease
NUM_DOMAINS = 2    # Domain A (Oxford), Domain B (Istanbul)
