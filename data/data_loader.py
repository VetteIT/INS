"""
Data loader modul - vytvara PyTorch DataLoadery pre nase datasety.
Hlavna funkcia je create_data_loaders ktora pripravi data pre trenovanie.

Tu robime aj rozdelenie na train/test podla speakerov (aby ten isty
clovek nebol aj v train aj v test - to by bolo podvadzanie).

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import GroupKFold

from config.settings import TRAINING_CONFIG
from data.pcgita_dataset import PCGITADataset
from data.neurovoz_dataset import NeurovozDataset
from data.pdita_dataset import PDITADataset


def get_domain_datasets(domain_name, transform=None, feature_type="raw"):
    """
    Vrati dataset objekt pre danu domenu.
    Toto je jednoducha factory funkcia - podla nazvu vrati spravny dataset.
    
    Parametre:
        domain_name (str): nazov domeny ("PC-GITA", "Neurovoz", "PDITA")
        transform: transformacie
        feature_type: typ features
        
    Vrati:
        dataset objekt
    """
    # Jednoduche mapovanie nazov -> trieda
    dataset_map = {
        "PC-GITA": PCGITADataset,
        "Neurovoz": NeurovozDataset,
        "PDITA": PDITADataset,
    }
    
    if domain_name not in dataset_map:
        raise ValueError(f"Neznama domena: {domain_name}. Moznosti: {list(dataset_map.keys())}")
    
    # Vytvorime dataset
    dataset_class = dataset_map[domain_name]
    dataset = dataset_class(transform=transform, feature_type=feature_type)
    
    return dataset


def create_data_loaders(dataset, test_size=0.2, batch_size=None):
    """
    Vytvori train a test DataLoader z datasetu.
    Rozdelenie robime podla speakerov (speaker-independent split),
    aby sme zabranili data leakage.
    
    Parametre:
        dataset: nas dataset objekt
        test_size (float): podiel dat na testovanie (0.2 = 20%)
        batch_size (int): velkost batchu (ak None, pouzije sa z configu)
        
    Vrati:
        tuple: (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]
    
    # Ziskame speaker IDs a labely
    speaker_ids = dataset.get_speaker_ids()
    labels = dataset.get_labels()
    
    # Ak nemame speaker IDs, urobime nahodne rozdelenie
    if len(speaker_ids) == 0 or len(set(speaker_ids)) <= 1:
        print("  Nemame speaker IDs, robime nahodne rozdelenie...")
        return _random_split(dataset, test_size, batch_size)
    
    # Speaker-independent split pomocou GroupKFold
    return _speaker_split(dataset, speaker_ids, test_size, batch_size)


def _speaker_split(dataset, speaker_ids, test_size, batch_size):
    """
    Rozdeli dataset podla speakerov.
    Kazdy speaker je bud v train alebo v test, nikdy v oboch.
    To je dolezite aby model nevidel toho isteho cloveka v train aj test.
    """
    # Pouzijeme GroupKFold s k=5 (20% test)
    n_splits = max(2, int(1.0 / test_size))
    
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Potrebujeme numpy arrays
    X = np.arange(len(dataset))
    y = np.array(dataset.get_labels())
    groups = np.array(speaker_ids)
    
    # Vezmeme prvy split
    for train_idx, test_idx in group_kfold.split(X, y, groups):
        break  # vezmeme len prvy fold
    
    # Vytvorime subset datasety
    train_subset = Subset(dataset, train_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())
    
    print(f"  Speaker-independent split: train={len(train_subset)}, test={len(test_subset)}")
    
    # Vytvorime DataLoadery
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # na Windows moze byt problem s num_workers > 0
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    
    return train_loader, test_loader


def _random_split(dataset, test_size, batch_size):
    """
    Nahodne rozdeli dataset na train a test.
    Pouzijeme ked nemame info o speakeroch.
    """
    n = len(dataset)
    n_test = int(n * test_size)
    n_train = n - n_test
    
    # Nahodne indexy
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_subset = Subset(dataset, train_idx.tolist())
    test_subset = Subset(dataset, test_idx.tolist())
    
    print(f"  Nahodne rozdelenie: train={len(train_subset)}, test={len(test_subset)}")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    
    return train_loader, test_loader


def create_domain_adaptation_loaders(source_dataset, target_dataset, batch_size=None):
    """
    Vytvori DataLoadery pre domain adaptation.
    Source domena ma labely, target domena nema (unsupervised DA).
    
    Parametre:
        source_dataset: zdrojovy dataset (s labelmi)
        target_dataset: cielovy dataset (bez labelov pri DA)
        batch_size: velkost batchu
        
    Vrati:
        tuple: (source_loader, target_loader)
    """
    if batch_size is None:
        batch_size = TRAINING_CONFIG["batch_size"]
    
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # drop last aby boli batche rovnako velke
    )
    
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    return source_loader, target_loader
