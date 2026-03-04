"""
Tradicne ML klasifikatory pre detekciu Parkinsonovej choroby.
Tieto modely pracuju s extrahovanymi akustickymi prizvukmi (MFCC, jitter, shimmer...).
Nepracuju priamo so surovou recou ani so spektrogramami.

Implementujeme dva klasifikatory:
1. SVM (Support Vector Machine) - klasicky ML model
2. MLP (Multi-Layer Perceptron) - jednoducha NN v sklearn

Pre domain adaptation pouzivame MLP v PyTorchu (aby sme mali gradient).

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config.settings import SVM_CONFIG, MLP_CONFIG


class SVMClassifier:
    """
    SVM klasifikator pre PD detekciu.
    Pouzivame sklearn SVC s RBF kernelom.
    SVM je klasicky ML model ktory hlada optimalnu separacnu rovinu.
    
    Poznamka: SVM nepodporuje gradient, takze ho nemozeme pouzit
    priamo s domain adaptation technikami. Pouzivame ho len ako baseline.
    """

    def __init__(self):
        """Inicializacia SVM modelu s preprocessing pipeline."""
        
        # Pipeline: StandardScaler (normalizacia) -> SVM
        # StandardScaler normalizuje features na priemer=0, std=1
        # To je dolezite pretoze SVM je citlive na skalu features
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel=SVM_CONFIG["kernel"],
                C=SVM_CONFIG["C"],
                gamma=SVM_CONFIG["gamma"],
                probability=True,  # chceme aj pravdepodobnosti, nie len label
                random_state=42,
            ))
        ])
        
        self.is_fitted = False  # ci uz bol model natrenovany

    def fit(self, X, y):
        """
        Natrénuje SVM model na danych datach.
        
        Parametre:
            X (numpy array): features [n_samples, n_features]
            y (numpy array): labely [n_samples]
        """
        print(f"  Trenujem SVM na {X.shape[0]} vzorkach, {X.shape[1]} prizvukoch...")
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"  SVM natrenovane!")

    def predict(self, X):
        """
        Predikcie pomocou natrenovaneho modelu.
        
        Parametre:
            X: features
        
        Vrati:
            numpy array: predikovane labely
        """
        if not self.is_fitted:
            raise RuntimeError("SVM este nebol natrenovany! Zavolajte najprv fit().")
        
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predikcia pravdepodobnosti.
        
        Vrati:
            numpy array: pravdepodobnosti pre kazdu triedu [n_samples, n_classes]
        """
        if not self.is_fitted:
            raise RuntimeError("SVM este nebol natrenovany!")
        
        return self.model.predict_proba(X)


class MLPTraditionalClassifier(nn.Module):
    """
    MLP (Multi-Layer Perceptron) klasifikator v PyTorchu.
    Toto je jednoducha plne prepojena neuronova siet.
    
    Pouzivame PyTorch aby sme mohli robit domain adaptation
    (potrebujeme gradient pre backpropagation).
    
    Architektura:
        Input -> FC(256) -> ReLU -> Dropout
              -> FC(128) -> ReLU -> Dropout
              -> FC(64)  -> ReLU -> Dropout
              -> FC(2)   -> Output
    """

    def __init__(self, input_dim, num_classes=None, hidden_sizes=None, dropout_rate=None):
        """
        Parametre:
            input_dim (int): velkost vstupneho feature vektora
            num_classes (int): pocet tried
            hidden_sizes (list): velkosti skrytych vrstiev
            dropout_rate (float): dropout rate
        """
        super(MLPTraditionalClassifier, self).__init__()
        
        if num_classes is None:
            num_classes = MLP_CONFIG["num_classes"]
        if hidden_sizes is None:
            hidden_sizes = MLP_CONFIG["hidden_sizes"]
        if dropout_rate is None:
            dropout_rate = MLP_CONFIG["dropout_rate"]
        
        self.input_dim = input_dim
        self.feature_dim = hidden_sizes[-1]  # posledna skryta vrstva = feature dim
        
        # Vytvorime skryte vrstvy dynamicky
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Posledna vrstva pre klasifikaciu
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)

    def extract_features(self, x):
        """
        Extrahuje feature vektor (vystup zo skrytych vrstiev).
        
        Parametre:
            x (torch.Tensor): vstupne features [batch, input_dim]
        
        Vrati:
            torch.Tensor: feature vektor [batch, feature_dim]
        """
        features = self.feature_extractor(x)
        return features

    def forward(self, x):
        """
        Dopredny priechod celym modelom.
        
        Parametre:
            x (torch.Tensor): vstupne features [batch, input_dim]
        
        Vrati:
            torch.Tensor: logity [batch, num_classes]
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def get_feature_dim(self):
        """Vrati velkost feature vektora."""
        return self.feature_dim
