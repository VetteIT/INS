"""
Konfigurácia projektu - všetky hyperparametre na jednom mieste.
Vzor: Cvičenie 4 - FF.ipynb (hyperparametre ako premenné na začiatku kódu)

Semestrálny projekt: Porovnanie domain adaptation techník
pre detekciu Parkinsonovej choroby z reči
"""

import os

import torch

# ---- Zariadenie (GPU/CPU) ----
# Vzor z Cvičenie 3 - Komponenty neurónových sietí.ipynb
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Cesty ----
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# ---- Hyperparametre trénovania ----
# Vzor z Cvičenie 4 - FF.ipynb: batch_size, learning_rate, num_epochs
BATCH_SIZE = 32
BATCH_SIZE_DA = 64      # väčší batch pre stabilnejší odhad MMD/CORAL
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DA_EPOCHS = 100         # viac epoch pre domain adaptation
DA_LR = 0.0005
RANDOM_SEED = 42

# ---- Veľkosti vrstiev ----
# Vzor z Cvičenie 3: parameterizovaný návrh sietí
HIDDEN_SIZE = 128       # väčšia skrytá vrstva pre lepšiu reprezentačnú kapacitu
FEATURE_DIM = 64        # dimenzia spoločného feature space pre DA

# ---- Regularizácia ----
DROPOUT_RATE = 0.3
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4

# ---- Domain adaptácia ----
# DANN: Ganin & Lempitsky (2015) https://arxiv.org/abs/1409.7495
DANN_LAMBDA = 0.5

# MMD: Gretton et al. (2012) https://jmlr.org/papers/v13/gretton12a.html
MMD_LAMBDA = 0.5

# CORAL: Sun & Saenko (2016) https://arxiv.org/abs/1607.01719
CORAL_LAMBDA = 1.0

# CDAN: Long et al. (2018) https://arxiv.org/abs/1705.10667
# Conditional Adversarial Domain Adaptation (class-conditional DANN)
CDAN_LAMBDA = 0.5

# Contrastive DA: prototype-based cross-domain alignment
# Ref: Yang et al. (2021) "Cross-domain Contrastive Learning for UDA"
CONTRASTIVE_LAMBDA = 0.5
CONTRASTIVE_TEMP = 0.07      # temperature for NT-Xent / InfoNCE loss
CONTRASTIVE_CONF = 0.75      # min pseudo-label confidence for target samples

# ---- Bootstrap štatistika ----
N_BOOTSTRAP = 1000           # počet bootstrap iterácií pre CI

# ---- Dáta ----
# 12 spoločných akustických príznakov (Oxford ↔ Istanbul)
NUM_FEATURES = 12
# 47 rozšírených príznakov (Istanbul only: MFCC, Delta, GNE, multi-band HNR)
NUM_FEATURES_FULL = 47
NUM_CLASSES = 2    # 0 = Healthy, 1 = Parkinson's Disease
NUM_DOMAINS = 4    # A=Oxford, B=Istanbul-R1, C=Istanbul-R2, D=Istanbul-R3
