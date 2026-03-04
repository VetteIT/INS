"""
Spolocna feature extractor siet.
Tato siet extrahuje features z roznych typov vstupov a
pouziva sa ako zaklad pre domain adaptation techniky.

Idea je taka ze feature extractor sa nauci takú reprezentaciu,
ktora je uzitocna pre klasifikaciu ale zaroven invariantna voci domene.
(t.j. features z PC-GITA a Neurovoz by mali vyzerat podobne)

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
import torch.nn as nn

from config.settings import CNN_CONFIG, MLP_CONFIG


class FeatureExtractorNetwork(nn.Module):
    """
    Univerzalny feature extractor ktory mozeme pouzit s Lubovolnym
    domain adaptation pristupom.
    
    Funguje s roznym typmi vstupov:
    - spektrogram (2D) -> CNN cast
    - akusticke prizvuky (1D) -> FC cast
    """

    def __init__(self, input_type="spectrogram", input_dim=None, feature_dim=256):
        """
        Parametre:
            input_type (str): "spectrogram" alebo "features"
            input_dim (int): velkost vstupu (pre "features" typ)
            feature_dim (int): velkost vystupneho feature vektora
        """
        super(FeatureExtractorNetwork, self).__init__()
        
        self.input_type = input_type
        self.feature_dim = feature_dim
        
        if input_type == "spectrogram":
            # CNN pre spektrogramy
            self.backbone = self._build_cnn_backbone()
            self.fc = nn.Sequential(
                nn.Linear(256, feature_dim),
                nn.ReLU(),
                nn.Dropout(CNN_CONFIG.get("dropout_rate", 0.5)),
            )
        elif input_type == "features":
            # FC siet pre akusticke features
            if input_dim is None:
                raise ValueError("Pre 'features' typ musite zadat input_dim!")
            
            dropout_rate = MLP_CONFIG.get("dropout_rate", 0.3)
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
            self.fc = nn.Sequential(
                nn.Linear(256, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        else:
            raise ValueError(f"Neznamy input_type: {input_type}")

    def _build_cnn_backbone(self):
        """
        Vytvori CNN backbone pre spektrogramy.
        Podobna architektura ako CNNClassifier ale bez poslednej vrstvy.
        """
        backbone = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
        )
        return backbone

    def forward(self, x):
        """
        Extrahuje features z vstupu.
        
        Parametre:
            x: vstupne data (spektrogram alebo feature vektor)
        
        Vrati:
            torch.Tensor: feature vektor [batch, feature_dim]
        """
        if self.input_type == "spectrogram":
            x = self.backbone(x)
            x = x.view(x.size(0), -1)  # flatten
        else:
            x = self.backbone(x)
        
        features = self.fc(x)
        return features

    def get_feature_dim(self):
        """Vrati velkost feature vektora."""
        return self.feature_dim
