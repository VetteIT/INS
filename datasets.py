"""
Týždeň 2: Vlastné Dataset triedy a DataLoader.
Vzor: Cvičenie 2 - Práca s dátami.ipynb (trieda Dataset, DataLoader)

Vytvárame vlastný Dataset pre naše dáta, podobne ako v cvičení 2
kde sme robili Days_set a Numbers_set triedy.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import DATA_DIR, BATCH_SIZE, RANDOM_SEED


class ParkinsonDataset(Dataset):
    """
    Vlastný Dataset pre akustické príznaky Parkinsonovej choroby.
    Vzor: Cvičenie 2 - triedy Dataset s __len__ a __getitem__
    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, features, labels):
        # Konverzia na tensory - vzor z Cvičenie 1 - Tensory.ipynb
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_class_weights(y):
    """
    Výpočet váh tried pre vyváženú loss funkciu.
    Dôležité pre nevyvážené datasety (Oxford: 75% PD / 25% Healthy).
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    """
    classes = np.unique(y)
    n_samples = len(y)
    weights = []
    for c in classes:
        count = np.sum(y == c)
        weights.append(n_samples / (len(classes) * count))
    return torch.tensor(weights, dtype=torch.float32)


def load_domain(domain_name):
    """Načíta CSV súbor pre danú doménu."""
    path = os.path.join(DATA_DIR, f'{domain_name}.csv')
    df = pd.read_csv(path)

    # Oddelenie príznakov a labelov
    feature_cols = [c for c in df.columns if c not in ('label', 'domain')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    return X, y


def create_loaders(X, y, test_size=0.2):
    """
    Rozdelí dáta na trénovaciu a testovaciu sadu, normalizuje a vytvorí DataLoader.
    Vzor: Cvičenie 2 - DataLoader s batch_size a shuffle

    Args:
        X: príznaky (numpy array)
        y: labely (numpy array)
        test_size: pomer testovacej sady

    Returns:
        train_loader, test_loader, scaler
    """
    # Rozdelenie na trénovaciu a testovaciu sadu
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    # Normalizácia (fit len na trénovacej sade!)
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Vytvorenie datasetov a loaderov
    # Vzor: Cvičenie 2 - DataLoader(dataset, batch_size, shuffle)
    train_dataset = ParkinsonDataset(X_train, y_train)
    test_dataset = ParkinsonDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader, scaler


def create_cross_domain_loaders(X_source, y_source, X_target, y_target):
    """
    Vytvorí loadery pre cross-domain experimenty.
    Trénovanie na celom zdrojovom datasete, testovanie na celom cieľovom.

    DÔLEŽITÉ: Každá doména sa normalizuje NEZÁVISLE (vlastný scaler).
    Toto je štandard v domain adaptation literatúre, pretože rôzne
    nemocnice/datasety majú rôzne rozsahy meraní.

    Ref: Ganin et al. (2015) - "features are standardized per domain"
    """
    # Normalizácia každej domény zvlášť
    source_scaler = StandardScaler()
    target_scaler = StandardScaler()
    X_source = source_scaler.fit_transform(X_source)
    X_target = target_scaler.fit_transform(X_target)

    source_dataset = ParkinsonDataset(X_source, y_source)
    target_dataset = ParkinsonDataset(X_target, y_target)

    source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return source_loader, target_loader, source_scaler
