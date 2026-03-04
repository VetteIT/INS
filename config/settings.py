"""
Hlavny konfiguracny subor pre cely projekt.
Tu su vsetky nastavenia - cesty k datasetom, hyperparametre modelov,
nastavenia trenovania a domain adaptation.

Autori: Dmytro Protsun, Mykyta Olym
Predmet: INS - Semestrálny projekt
Tema: Porovnanie domain adaptation techník pre detekciu Parkinsonovej choroby z reči
"""

import os

# ============================================================
# CESTY K DATASETOM
# Kazdy dataset je ulozeny v svojom priecinku
# Musite si stiahnut datasety a dat ich do spravnych priecinkov
# ============================================================

# Zakladny priecinok kde su vsetky data
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

# Cesty k jednotlivym datasetom
PCGITA_DIR = os.path.join(BASE_DATA_DIR, "PC-GITA")
NEUROVOZ_DIR = os.path.join(BASE_DATA_DIR, "Neurovoz")
PDITA_DIR = os.path.join(BASE_DATA_DIR, "PDITA")

# Priecinok na ukladanie vysledkov a grafov
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")

# ============================================================
# NASTAVENIA PRE AUDIO PREPROCESSING
# Tieto hodnoty sme zvolili podla odporucani z clankov
# ============================================================

# Vzorkovacia frekvencia - 16kHz je standard pre recove ulohy
SAMPLE_RATE = 16000

# Dlzka segmentu v sekundach (kazdy zvuk rozrezeme na taketo kusky)
SEGMENT_LENGTH = 3.0

# Ci sa maju segmenty prekryvat
SEGMENT_OVERLAP = 0.5  # 50% prekrytie

# ============================================================
# NASTAVENIA PRE EXTRAKCIU PRIZVUKOV (FEATURES)
# ============================================================

# MFCC nastavenia
N_MFCC = 13          # pocet MFCC koeficientov
N_FFT = 512           # velkost FFT okna
HOP_LENGTH = 256      # hop length pre FFT
N_MELS = 128          # pocet mel filtrov pre spektrogram

# Jitter a Shimmer sa pocitaju z F0 (zakladna frekvencia)
F0_MIN = 75           # minimalna F0 v Hz
F0_MAX = 500          # maximalna F0 v Hz

# Spektrogram nastavenia
SPEC_HEIGHT = 224     # vyska spektrogramu v pixeloch (pre CNN)
SPEC_WIDTH = 224      # sirka spektrogramu v pixeloch (pre CNN)

# ============================================================
# NASTAVENIA MODELOV
# ============================================================

# CNN model nastavenia
CNN_CONFIG = {
    "input_channels": 1,        # spektrogram je 1-kanalovy (grayscale)
    "num_classes": 2,           # PD vs Healthy
    "dropout_rate": 0.5,        # dropout aby sme predisli overfittingu
    "learning_rate": 0.001,     # learning rate pre Adam optimizer
    "feature_dim": 256,         # velkost feature vektora z CNN
}

# SVM model nastavenia
SVM_CONFIG = {
    "kernel": "rbf",            # radial basis function kernel
    "C": 1.0,                   # regularizacny parameter
    "gamma": "scale",           # gamma parameter pre RBF
}

# MLP model nastavenia
MLP_CONFIG = {
    "hidden_sizes": [256, 128, 64],  # velkosti skrytych vrstiev
    "dropout_rate": 0.3,             # dropout
    "learning_rate": 0.001,          # learning rate
    "num_classes": 2,                # PD vs Healthy
}

# Wav2Vec2 model nastavenia
WAV2VEC_CONFIG = {
    "model_name": "facebook/wav2vec2-base",  # predtrenovany model
    "num_classes": 2,                         # PD vs Healthy
    "learning_rate": 0.00001,                 # mensie lr pretoze finetunujeme
    "feature_dim": 768,                       # velkost feature vektora z wav2vec2
    "freeze_feature_extractor": True,         # zmrazime feature extractor
    "dropout_rate": 0.3,                      # dropout pre klasifikacnu hlavu
}

# ============================================================
# NASTAVENIA TRENOVANIA
# ============================================================

TRAINING_CONFIG = {
    "batch_size": 32,           # velkost batchu
    "num_epochs": 50,           # pocet epoch
    "learning_rate": 0.001,     # zakladny learning rate
    "early_stopping_patience": 10,  # po kolkych epochach bez zlepsenia zastavime
    "weight_decay": 1e-4,       # L2 regularizacia
    "num_workers": 4,           # pocet workerov pre data loading
    "seed": 42,                 # seed pre reprodukovatelnost
}

# ============================================================
# NASTAVENIA DOMAIN ADAPTATION
# ============================================================

# DANN - Domain Adversarial Neural Network
DANN_CONFIG = {
    "lambda_domain": 1.0,       # vaha domain loss
    "alpha": 10.0,              # parameter pre gradient reversal
    "learning_rate": 0.001,
    "hidden_dim": 256,          # skryta vrstva domain classifiera
}

# MMD - Maximum Mean Discrepancy
MMD_CONFIG = {
    "lambda_mmd": 1.0,          # vaha MMD loss
    "kernel_type": "rbf",       # typ kernelu pre MMD
    "kernel_bandwidth": [0.01, 0.1, 1.0, 10.0, 100.0],  # bandwidths pre multi-kernel MMD
    "learning_rate": 0.001,
}

# Contrastive Domain Alignment
CONTRASTIVE_CONFIG = {
    "lambda_contrastive": 0.5,  # vaha contrastive loss
    "temperature": 0.07,        # temperatura pre contrastive loss
    "projection_dim": 128,      # velkost projekcie
    "learning_rate": 0.001,
}

# Multi-Source Domain Adaptation
MULTI_SOURCE_CONFIG = {
    "lambda_domain": 1.0,       # vaha domain loss pre kazdu source domenu
    "aggregation": "weighted",  # ako kombinujeme source domeny (weighted/average)
    "learning_rate": 0.001,
}

# ============================================================
# ZOZNAM EXPERIMENTOV KTORE CHCEME SPUSTIT
# ============================================================

# Nazvy datasetov ktore pouzivame ako domeny
DOMAIN_NAMES = ["PC-GITA", "Neurovoz", "PDITA"]

# Adaptacne techniky ktore chceme porovnat
ADAPTATION_METHODS = ["baseline", "dann", "mmd", "contrastive", "multi_source"]

# Typy modelov ktore chceme otestovat
MODEL_TYPES = ["cnn", "traditional", "wav2vec"]

# Metriky ktore chceme vyhodnotit
METRICS = ["accuracy", "auc", "f1_score", "sensitivity", "specificity"]
