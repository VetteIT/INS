"""
CNN klasifikator pre detekciu Parkinsonovej choroby.
Pouzivame konvolucnu neuronovu siet nad spektrogramami.
CNN je dobra na spracovanie obrazkov (a spektrogram je vlastne obrazok).

Architektura:
    Vstup -> Conv2D -> BatchNorm -> ReLU -> MaxPool
          -> Conv2D -> BatchNorm -> ReLU -> MaxPool
          -> Conv2D -> BatchNorm -> ReLU -> MaxPool
          -> Conv2D -> BatchNorm -> ReLU -> AdaptiveAvgPool
          -> Flatten -> FC -> Dropout -> FC -> Vystup (2 triedy)

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
import torch.nn as nn

from config.settings import CNN_CONFIG


class CNNClassifier(nn.Module):
    """
    CNN model na klasifikaciu spektrogramov.
    Vstupom je mel spektrogram a vystupom je predikcia PD/Healthy.
    """

    def __init__(self, num_classes=None, feature_dim=None, input_channels=None):
        """
        Inicializacia CNN modelu.
        
        Parametre:
            num_classes (int): pocet tried (2 = PD vs Healthy)
            feature_dim (int): velkost feature vektora (pred poslednou vrstvou)
            input_channels (int): pocet vstupnych kanalov (1 pre grayscale spektrogram)
        """
        super(CNNClassifier, self).__init__()
        
        # Nastavenia - pouzijeme z configu ak nie su zadane
        if num_classes is None:
            num_classes = CNN_CONFIG["num_classes"]
        if feature_dim is None:
            feature_dim = CNN_CONFIG["feature_dim"]
        if input_channels is None:
            input_channels = CNN_CONFIG["input_channels"]
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # ===== KONVOLUCNE VRSTVY =====
        # Kazda vrstva: Conv2D -> BatchNorm -> ReLU -> MaxPool
        # Postupne zvysujeme pocet filtrov: 32 -> 64 -> 128 -> 256
        
        # Prva konvolucna vrstva
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # zmensi rozlisenie na polovicu
        )
        
        # Druha konvolucna vrstva
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Tretia konvolucna vrstva
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Stvrta konvolucna vrstva
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling -> [batch, 256, 1, 1]
        )
        
        # ===== PLNE PREPOJENE VRSTVY =====
        # Feature vektor -> klasifikacia
        
        self.feature_layer = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(CNN_CONFIG["dropout_rate"]),
        )
        
        # Posledna vrstva pre klasifikaciu
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        """
        Extrahuje feature vektor z vstupneho spektrogramu.
        Toto pouzivame pri domain adaptation - chceme features bez klasifikacie.
        
        Parametre:
            x (torch.Tensor): vstupny spektrogram [batch, 1, H, W]
        
        Vrati:
            torch.Tensor: feature vektor [batch, feature_dim]
        """
        # Prechod cez konvolucne vrstvy
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten - z [batch, 256, 1, 1] na [batch, 256]
        x = x.view(x.size(0), -1)
        
        # Feature vrstva
        features = self.feature_layer(x)
        
        return features

    def forward(self, x):
        """
        Dopredny priechod celym modelom.
        
        Parametre:
            x (torch.Tensor): vstupny spektrogram [batch, 1, H, W]
        
        Vrati:
            torch.Tensor: logity pre kazdu triedu [batch, num_classes]
        """
        # Najprv extrahujeme features
        features = self.extract_features(x)
        
        # Potom klasifikacia
        logits = self.classifier(features)
        
        return logits

    def get_feature_dim(self):
        """Vrati velkost feature vektora."""
        return self.feature_dim
