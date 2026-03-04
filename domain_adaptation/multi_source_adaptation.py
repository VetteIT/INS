"""
Multi-Source Domain Adaptation.
Namiesto jednej source domeny pouzivame VIACERO source domen.
Napriklad: trénujeme na MDVR-KCL (ReadText + SpontaneousDialogue), testujeme na ItalianPVS.

Princip:
    Ked mame viacero source domen, mame viac dat na trenovanie
    a az model sa dokaze naučit z roznych distribucii.
    
    Implementujeme dva pristupy:
    1. Simple Mix - jednoducho kombinujeme data zo vsetkych source domen
    2. Weighted Mix - kazdej source domene priradime vahu podla toho
       ako je blizko target domene

    Navyse, pre kazdu source domenu mame vlastny domain classifier
    (ako v DANN) a minimalizujeme domain discrepancy voci target domene.

Paper: "Moment Matching for Multi-Source Domain Adaptation" (Peng et al., 2019)

Autori: Dmytro Protsun, Mykyta Olym
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config.settings import MULTI_SOURCE_CONFIG, TRAINING_CONFIG
from domain_adaptation.mmd_adaptation import compute_mmd


class MultiSourceTrainer:
    """
    Multi-Source Domain Adaptation Trener.
    
    Pouzivame viacero source domen pre trenovanie.
    Kazda source domena prispieva k celkovemu trenovaniu.
    """

    def __init__(self, feature_extractor, label_classifier, num_sources, device="cpu"):
        """
        Parametre:
            feature_extractor: spolocny feature extractor pre vsetky domeny
            label_classifier: klasifikator PD/Healthy
            num_sources: pocet source domen
            device: "cpu" alebo "cuda"
        """
        self.device = device
        self.num_sources = num_sources
        
        self.feature_extractor = feature_extractor.to(device)
        self.label_classifier = label_classifier.to(device)
        
        # Loss funkcie
        self.label_criterion = nn.CrossEntropyLoss()
        
        # Optimizer - len feature extractor a label classifier
        all_params = (
            list(self.feature_extractor.parameters()) +
            list(self.label_classifier.parameters())
        )
        
        self.optimizer = optim.Adam(
            all_params,
            lr=MULTI_SOURCE_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        
        # Historia
        self.train_losses = []
        self.domain_losses = []
        self.val_accuracies = []

    def train_epoch(self, source_loaders, target_loader):
        """
        Jedna trenovacia epocha s multi-source DA.
        
        Parametre:
            source_loaders: list DataLoaderov pre kazdu source domenu
            target_loader: DataLoader pre target domenu
        """
        self.feature_extractor.train()
        self.label_classifier.train()
        
        total_label_loss = 0.0
        total_domain_loss = 0.0
        num_batches = 0
        
        # Iteratory pre kazdu source a target domenu
        source_iters = [iter(loader) for loader in source_loaders]
        target_iter = iter(target_loader)
        
        # Pocet krokov = dlzka najdlhsieho source loadera
        max_steps = max(len(loader) for loader in source_loaders)
        
        for step in range(max_steps):
            # Ziskame batch z kazdej source domeny
            source_data_list = []
            source_labels_list = []
            
            for i, source_iter in enumerate(source_iters):
                try:
                    s_data, s_labels = next(source_iter)
                except StopIteration:
                    # Restartujeme iterator ak dosli data
                    source_iters[i] = iter(source_loaders[i])
                    s_data, s_labels = next(source_iters[i])
                
                source_data_list.append(s_data.to(self.device))
                source_labels_list.append(s_labels.to(self.device))
            
            # Target data
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            target_data = target_data.to(self.device)
            
            # ===== DOPREDNY PRIECHOD =====
            
            # Extrahujeme features z target domeny
            target_features = self.feature_extractor(target_data)
            
            # Pre kazdu source domenu
            weighted_label_loss = 0.0
            weighted_domain_loss = 0.0
            domain_weights = []  # ulozime vahy pre normalizaciu
            label_losses = []    # ulozime loss pre kazdu domenu
            
            for i in range(self.num_sources):
                # Features z source domeny
                source_features = self.feature_extractor(source_data_list[i])
                
                # Label klasifikacia
                label_output = self.label_classifier(source_features)
                label_loss = self.label_criterion(label_output, source_labels_list[i])
                
                # MMD medzi touto source a target domenou
                mmd_loss = compute_mmd(source_features, target_features)
                
                # Vahy pre tuto domenu
                if MULTI_SOURCE_CONFIG["aggregation"] == "weighted":
                    # Dynamicke vahy - blizsie domeny dostanu vyssiu vahu
                    domain_weight = 1.0 / (mmd_loss.item() + 1e-6)
                else:
                    # Rovnake vahy
                    domain_weight = 1.0
                
                domain_weights.append(domain_weight)
                label_losses.append(label_loss)
                weighted_domain_loss += mmd_loss
            
            # Normalizujeme vahy aby sucet bol 1
            weight_sum = sum(domain_weights)
            for i in range(self.num_sources):
                normalized_weight = domain_weights[i] / weight_sum
                weighted_label_loss += normalized_weight * label_losses[i]
            
            weighted_domain_loss = weighted_domain_loss / self.num_sources
            
            # Celkova loss
            total_loss = (weighted_label_loss +
                         MULTI_SOURCE_CONFIG["lambda_domain"] * weighted_domain_loss)
            
            # ===== SPATNY PRIECHOD =====
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_label_loss += weighted_label_loss.item()
            total_domain_loss += weighted_domain_loss.item()
            num_batches += 1
        
        avg_label_loss = total_label_loss / max(num_batches, 1)
        avg_domain_loss = total_domain_loss / max(num_batches, 1)
        
        return avg_label_loss, avg_domain_loss

    def evaluate(self, test_loader):
        """Vyhodnoti model na testovacich datach."""
        self.feature_extractor.eval()
        self.label_classifier.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                
                features = self.feature_extractor(batch_data)
                outputs = self.label_classifier(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        return {
            "accuracy": accuracy,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    def train(self, source_loaders, target_loader, val_loader=None, num_epochs=None):
        """Kompletny multi-source DA trenovaci cyklus."""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n=== Multi-Source DA trenovanie ({num_epochs} epoch, "
              f"{self.num_sources} source domen) ===")
        
        for epoch in range(num_epochs):
            label_loss, domain_loss = self.train_epoch(source_loaders, target_loader)
            self.train_losses.append(label_loss)
            self.domain_losses.append(domain_loss)
            
            val_info = ""
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_acc = val_results["accuracy"]
                self.val_accuracies.append(val_acc)
                val_info = f", Val Acc: {val_acc:.4f}"
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = {
                        'feature_extractor': copy.deepcopy(self.feature_extractor.state_dict()),
                        'label_classifier': copy.deepcopy(self.label_classifier.state_dict()),
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                    print(f"  Early stopping na epoche {epoch+1}")
                    break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epocha {epoch+1}/{num_epochs}: "
                      f"Label Loss = {label_loss:.4f}, "
                      f"Domain Loss = {domain_loss:.6f}{val_info}")
        
        # Nacitame najlepsi model ak mame
        if best_model_state is not None:
            self.feature_extractor.load_state_dict(best_model_state['feature_extractor'])
            self.label_classifier.load_state_dict(best_model_state['label_classifier'])
            print(f"  Najlepsia validacna presnost: {best_val_acc:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "domain_losses": self.domain_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_val_acc,
        }
