"""
Hlavny trenovaci modul ktory spaja vsetky casti dokopy.
Riadi celý proces trenovania - vytvori model, data loadery,
zvoli DA techniku a spusti trenovanie.

Toto je vlastne "mozog" celeho systemu - vsetko sa riadi odtialto.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import torch
import torch.nn as nn
import numpy as np

from config.settings import (
    TRAINING_CONFIG, CNN_CONFIG, MLP_CONFIG,
    RESULTS_DIR, MODELS_DIR
)
from models.cnn_classifier import CNNClassifier
from models.traditional_classifier import MLPTraditionalClassifier
from models.feature_extractor_net import FeatureExtractorNetwork
from domain_adaptation.baseline import BaselineTrainer
from domain_adaptation.dann import DANNTrainer
from domain_adaptation.mmd_adaptation import MMDTrainer
from domain_adaptation.contrastive_alignment import ContrastiveTrainer
from domain_adaptation.multi_source_adaptation import MultiSourceTrainer


def set_seed(seed=None):
    """
    Nastavi seed pre reprodukovatelne vysledky.
    Ked pouzijeme rovnaky seed, dostaneme rovnake vysledky.
    To je dolezite pre vedecku pracu.
    """
    if seed is None:
        seed = TRAINING_CONFIG["seed"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class _Wav2VecFeatureWrapper(nn.Module):
    """
    Wrapper okolo Wav2VecClassifier ktory vola len extract_features.
    Toto potrebujeme pre DA techniky - chceme features, nie cele logity.
    """

    def __init__(self, wav2vec_model):
        super(_Wav2VecFeatureWrapper, self).__init__()
        self.model = wav2vec_model

    def forward(self, x):
        """Vrati features namiesto logitov."""
        return self.model.extract_features(x)

    def get_feature_dim(self):
        """Vrati velkost feature vektora."""
        return self.model.get_feature_dim()


class ModelTrainer:
    """
    Hlavny trener ktory riadi cely experiment.
    Vytvori modely, zvoli DA techniku a spusti trenovanie.
    """

    def __init__(self, model_type="cnn", adaptation_method="baseline",
                 input_dim=None, device=None):
        """
        Parametre:
            model_type (str): typ modelu - "cnn", "traditional", "wav2vec"
            adaptation_method (str): DA technika - "baseline", "dann", "mmd",
                                     "contrastive", "multi_source"
            input_dim (int): velkost vstupu (pre traditional model)
            device (str): "cpu" alebo "cuda"
        """
        # Nastavime seed
        set_seed()
        
        # Automaticky vyberieme zariadenie
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Pouzivam zariadenie: {self.device}")
        
        self.model_type = model_type
        self.adaptation_method = adaptation_method
        self.input_dim = input_dim
        
        # Vytvorime model a DA trener
        self._create_model_and_trainer()
        
        # Vytvorime priecinky na vysledky
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

    def _create_model_and_trainer(self):
        """
        Vytvori model a DA trener podla konfigurácie.
        Toto je trošku zlozitejsie pretoze musime spravne poskladat
        feature extractor + classifier + DA techniku.
        """
        if self.model_type == "cnn":
            # Pre CNN pouzivame spektrogramy ako vstup
            if self.adaptation_method == "baseline":
                # Jednoduchy CNN bez DA
                self.model = CNNClassifier()
                self.trainer = BaselineTrainer(self.model, self.device)
            else:
                # Pre DA techniky potrebujeme oddeleny feature extractor a classifier
                self.feature_extractor = FeatureExtractorNetwork(
                    input_type="spectrogram",
                    feature_dim=CNN_CONFIG["feature_dim"],
                )
                self.label_classifier = nn.Linear(
                    CNN_CONFIG["feature_dim"],
                    CNN_CONFIG["num_classes"],
                )
                self._create_da_trainer()
        
        elif self.model_type == "traditional":
            # Pre tradicne modely pouzivame akusticke prizvuky
            if self.input_dim is None:
                # Predpokladana velkost feature vektora
                # 13 MFCC * 8 (mean, std, min, max, delta_mean, delta_std, delta2_mean, delta2_std) = 104
                # + 4 F0 stats + jitter + shimmer + HNR + 10 spectral features = 121
                self.input_dim = 121
            
            if self.adaptation_method == "baseline":
                self.model = MLPTraditionalClassifier(input_dim=self.input_dim)
                self.trainer = BaselineTrainer(self.model, self.device)
            else:
                self.feature_extractor = FeatureExtractorNetwork(
                    input_type="features",
                    input_dim=self.input_dim,
                    feature_dim=MLP_CONFIG["hidden_sizes"][-1],
                )
                self.label_classifier = nn.Linear(
                    MLP_CONFIG["hidden_sizes"][-1],
                    MLP_CONFIG["num_classes"],
                )
                self._create_da_trainer()
        
        elif self.model_type == "wav2vec":
            # Wav2Vec2 model - zatial len baseline a DANN
            from models.wav2vec_classifier import Wav2VecClassifier
            
            if self.adaptation_method == "baseline":
                self.model = Wav2VecClassifier()
                self.trainer = BaselineTrainer(self.model, self.device)
            else:
                # Pre DA s wav2vec pouzijeme wav2vec ako feature extractor
                # Vytvorime wrapper ktory vola extract_features (nie forward)
                self.model = Wav2VecClassifier()
                self.feature_extractor = _Wav2VecFeatureWrapper(self.model).to(self.device)
                self.label_classifier = nn.Linear(
                    self.model.get_feature_dim(),
                    2,
                )
                self._create_da_trainer()
        
        else:
            raise ValueError(f"Neznamy model typ: {self.model_type}")

    def _create_da_trainer(self):
        """Vytvori DA trener podla zvolenej metody."""
        if self.adaptation_method == "dann":
            self.trainer = DANNTrainer(
                self.feature_extractor,
                self.label_classifier,
                self.device,
            )
        elif self.adaptation_method == "mmd":
            self.trainer = MMDTrainer(
                self.feature_extractor,
                self.label_classifier,
                self.device,
            )
        elif self.adaptation_method == "contrastive":
            self.trainer = ContrastiveTrainer(
                self.feature_extractor,
                self.label_classifier,
                self.device,
            )
        elif self.adaptation_method == "multi_source":
            # Pre multi-source potrebujeme vediet pocet source domen
            # Vytvorime s predvolenym poctom 1, prestavime v train_multi_source()
            self.label_classifier = self.label_classifier.to(self.device)
            self.trainer = MultiSourceTrainer(
                self.feature_extractor,
                self.label_classifier,
                num_sources=1,
                device=self.device,
            )
        else:
            raise ValueError(f"Neznama DA metoda: {self.adaptation_method}")

    def train_baseline(self, train_loader, val_loader=None, num_epochs=None):
        """
        Natrénuje model bez domain adaptation.
        
        Parametre:
            train_loader: DataLoader s trenovacimi datami
            val_loader: DataLoader s validacnymi datami
            num_epochs: pocet epoch
        """
        print(f"\n{'='*60}")
        print(f"Trenovanie: {self.model_type} + baseline")
        print(f"{'='*60}")
        
        history = self.trainer.train(train_loader, val_loader, num_epochs)
        return history

    def train_with_da(self, source_loader, target_loader, val_loader=None, num_epochs=None):
        """
        Natrénuje model s domain adaptation.
        
        Parametre:
            source_loader: DataLoader so source datami
            target_loader: DataLoader s target datami
            val_loader: DataLoader s validacnymi datami
            num_epochs: pocet epoch
        """
        print(f"\n{'='*60}")
        print(f"Trenovanie: {self.model_type} + {self.adaptation_method}")
        print(f"{'='*60}")
        
        history = self.trainer.train(source_loader, target_loader, val_loader, num_epochs)
        return history

    def train_multi_source(self, source_loaders, target_loader, val_loader=None, num_epochs=None):
        """
        Natrénuje model s multi-source domain adaptation.
        
        Parametre:
            source_loaders: list DataLoaderov pre source domeny
            target_loader: DataLoader s target datami
            val_loader: DataLoader s validacnymi datami
        """
        print(f"\n{'='*60}")
        print(f"Trenovanie: {self.model_type} + multi_source ({len(source_loaders)} zdroje)")
        print(f"{'='*60}")
        
        # Aktualizujeme multi-source trainer s aktualnym poctom source domen
        self.trainer = MultiSourceTrainer(
            self.feature_extractor,
            self.label_classifier,
            num_sources=len(source_loaders),
            device=self.device,
        )
        
        history = self.trainer.train(source_loaders, target_loader, val_loader, num_epochs)
        return history

    def evaluate(self, test_loader):
        """
        Vyhodnoti model na testovacich datach.
        
        Parametre:
            test_loader: DataLoader s testovacimi datami
        
        Vrati:
            dict: vysledky evaluacie
        """
        results = self.trainer.evaluate(test_loader)
        return results

    def save_model(self, filename):
        """Ulozi model na disk."""
        filepath = os.path.join(MODELS_DIR, filename)
        
        if self.adaptation_method == "baseline":
            torch.save(self.model.state_dict(), filepath)
        else:
            # Ulozime feature extractor aj classifier
            torch.save({
                "feature_extractor": self.feature_extractor.state_dict(),
                "label_classifier": self.label_classifier.state_dict(),
            }, filepath)
        
        print(f"  Model ulozeny: {filepath}")

    def load_model(self, filename):
        """Nacita model z disku."""
        filepath = os.path.join(MODELS_DIR, filename)
        
        if self.adaptation_method == "baseline":
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
            self.label_classifier.load_state_dict(checkpoint["label_classifier"])
        
        print(f"  Model nacitany: {filepath}")
