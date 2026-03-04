"""
Wav2Vec2 klasifikator pre detekciu Parkinsonovej choroby.
Pouzivame predtrenovany Wav2Vec2 model od Facebooku (Meta).
Wav2Vec2 je foundation model ktory bol natrenovany na obrovskom
mnozstve audio dat a vie extrahovat kvalitne reprezentacie reci.

Princip:
1. Wav2Vec2 spracuje surovu rec (raw waveform)
2. Vytvori vysoko-urovnove features
3. My pridame klasifikacnu hlavu (Linear vrstvu)
4. Finetunujeme na nasich datach (PD vs Healthy)

Vyhoda oproti CNN: Nevyzaduje rucnu extrakciu features ani spektrogram.
Model sa sam nauci co je dolezite v reci.

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
import torch.nn as nn

from config.settings import WAV2VEC_CONFIG

# Skusime importovat transformers - ak nie je nainstalovana, dame vediet
try:
    from transformers import Wav2Vec2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("VAROVANIE: Kniznica 'transformers' nie je nainstalovana!")
    print("Pre Wav2Vec2 model spustite: pip install transformers")


class Wav2VecClassifier(nn.Module):
    """
    Klasifikator zalozeny na Wav2Vec2.
    Pouzivame predtrenovany Wav2Vec2 a pridame vlastnu klasifikacnu hlavu.
    
    Architektura:
        Raw audio -> Wav2Vec2 (predtrenovany) -> Pool -> FC -> Dropout -> FC -> Output
    """

    def __init__(self, num_classes=None, feature_dim=None, freeze_extractor=None):
        """
        Parametre:
            num_classes (int): pocet tried
            feature_dim (int): velkost feature vektora z wav2vec2
            freeze_extractor (bool): ci zmrazit feature extractor wav2vec2
        """
        super(Wav2VecClassifier, self).__init__()
        
        if num_classes is None:
            num_classes = WAV2VEC_CONFIG["num_classes"]
        if feature_dim is None:
            feature_dim = WAV2VEC_CONFIG["feature_dim"]
        if freeze_extractor is None:
            freeze_extractor = WAV2VEC_CONFIG["freeze_feature_extractor"]
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Kniznica 'transformers' nie je dostupna! Nainstalujte ju.")
        
        # Nacitame predtrenovany Wav2Vec2 model
        print(f"  Nacitavam predtrenovany model: {WAV2VEC_CONFIG['model_name']}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC_CONFIG["model_name"])
        
        # Zmrazime feature extractor ak treba
        # (trénujeme len transformer cast a nasu klasifikacnu hlavu)
        if freeze_extractor:
            print("  Zmrazujem feature extractor...")
            self.wav2vec2.feature_extractor._freeze_parameters()
        
        # Klasifikacna hlava
        # Wav2Vec2 vystup ma velkost 768 (pre base model)
        dropout_rate = WAV2VEC_CONFIG.get("dropout_rate", 0.3)
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Feature dimenzia po classification head
        self.output_feature_dim = 128
        
        # Posledna vrstva
        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, x):
        """
        Extrahuje features z rawvaveformu pomocou Wav2Vec2.
        
        Parametre:
            x (torch.Tensor): surovy audio signal [batch, length]
        
        Vrati:
            torch.Tensor: feature vektor [batch, output_feature_dim]
        """
        # Wav2Vec2 ocakava vstup [batch, length]
        # Ak ma channel dimenziu, odstranime ju
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Prechod cez Wav2Vec2
        outputs = self.wav2vec2(x)
        
        # Vezmeme posledny skryty stav
        hidden_states = outputs.last_hidden_state  # [batch, time_steps, 768]
        
        # Global average pooling cez casovu os
        pooled = torch.mean(hidden_states, dim=1)  # [batch, 768]
        
        # Prechod cez nasu classification head
        features = self.classification_head(pooled)  # [batch, 128]
        
        return features

    def forward(self, x):
        """
        Dopredny priechod celym modelom.
        
        Parametre:
            x (torch.Tensor): surovy audio signal [batch, length]
        
        Vrati:
            torch.Tensor: logity [batch, num_classes]
        """
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def get_feature_dim(self):
        """Vrati velkost feature vektora."""
        return self.output_feature_dim
