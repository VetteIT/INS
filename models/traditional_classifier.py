"""
Tradicne ML klasifikatory pre detekciu Parkinsonovej choroby.
Tieto modely pracuju s extrahovanymi akustickymi prizvukmi (MFCC, jitter, shimmer...).
Nepracuju priamo so surovou recou ani so spektrogramami.

Implementujeme MLP (Multi-Layer Perceptron) v PyTorchu,
aby sme mohli robit domain adaptation (potrebujeme gradient).

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
import torch.nn as nn

from config.settings import MLP_CONFIG


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
