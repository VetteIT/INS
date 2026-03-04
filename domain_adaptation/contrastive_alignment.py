"""
Contrastive Domain Alignment pre domain adaptation.
Pouzivame contrastive learning na zarovnanie (alignment) 
feature distribucii medzi source a target domenami.

Princip:
    - Vzorky z rovnakej triedy (aj z roznych domen) by mali mat
      PODOBNE features (small distance)
    - Vzorky z roznych tried by mali mat ODLISNE features (large distance)
    
    Pouzivame supervised contrastive loss na source datach
    a self-supervised contrastive alignment medzi domenami.

    Teplota (temperature) kontroluje "ostrost" distribúcie -
    mensia teplota = viac sa rozlisuje medzi pozitivnymi a negativnymi parmi.

Paper: Inspirovana "Supervised Contrastive Learning" (Khosla et al., 2020)

Autori: Dmytro Protsun, Mykyta Olym
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from config.settings import CONTRASTIVE_CONFIG, TRAINING_CONFIG


class ProjectionHead(nn.Module):
    """
    Projekcna hlavicka pre contrastive learning.
    Mapuje features do normalizeného priestoru kde sa pocita contrastive loss.
    
    Architektura: Linear -> ReLU -> Linear -> L2 normalizacia
    """
    
    def __init__(self, input_dim, projection_dim=128):
        """
        Parametre:
            input_dim: velkost vstupneho feature vektora
            projection_dim: velkost projekcie (typicky 128)
        """
        super(ProjectionHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )
    
    def forward(self, x):
        """Projekcia + L2 normalizacia."""
        projected = self.net(x)
        # L2 normalizacia - vsetky vektory budu mat dlzku 1
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised Contrastive Loss.
    Vzorky z rovnakej triedy su "pozitivne pary" (maju byt blizko).
    Vzorky z roznych tried su "negativne pary" (maju byt daleko).
    
    Parametre:
        features (torch.Tensor): normalizovane features [batch, dim]
        labels (torch.Tensor): labely [batch]
        temperature (float): teplota (mensia = ostrejsie)
    
    Vrati:
        torch.Tensor: contrastive loss (scalar)
    """
    batch_size = features.shape[0]
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=features.device)
    
    # Similarity matica (cosine similarity pretoze features su normalizovane)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Maska pre pozitivne pary (rovnaka trieda)
    labels = labels.view(-1, 1)
    mask_positive = (labels == labels.T).float()
    
    # Odstranime diagonaluy (sama so sebou)
    mask_self = torch.eye(batch_size, device=features.device)
    mask_positive = mask_positive - mask_self
    
    # Pre numericku stabilitu odcitame maximum
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Exponenty (okrem diagonaly)
    exp_logits = torch.exp(logits) * (1 - mask_self)
    
    # Log sum exp (denominator)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
    
    # Priemer log pravdepodobnosti pre pozitivne pary
    # (t.j. pre kazdu vzorku spocitame priemer log_prob cez vsetky pozitivne pary)
    num_positive = mask_positive.sum(dim=1)
    
    # Ak vzorka nema ziadny pozitivny par, preskocime ju
    valid_mask = num_positive > 0
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    
    mean_log_prob = (mask_positive * log_prob).sum(dim=1) / (num_positive + 1e-6)
    
    # Loss = -priemer cez vsetky validne vzorky
    loss = -mean_log_prob[valid_mask].mean()
    
    return loss


def domain_contrastive_loss(source_features, target_features, temperature=0.07):
    """
    Domain Contrastive Loss.
    Zarovnava feature priestory medzi source a target domenou.
    Snazi sa aby features z oboch domen boli v podobnom priestore.
    
    Parametre:
        source_features: features z source domeny [n_s, dim]
        target_features: features z target domeny [n_t, dim]
        temperature: teplota
    
    Vrati:
        torch.Tensor: domain contrastive loss
    """
    # Spojime features z oboch domen
    all_features = torch.cat([source_features, target_features], dim=0)
    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    n_total = n_source + n_target
    
    # Similarity matica
    similarity = torch.matmul(all_features, all_features.T) / temperature
    
    # Vytvorime "pseudo-labels" - chceme aby source a target
    # s podobnymi features boli blizko seba
    # Pouzijeme nearest-neighbor priradenie
    
    # Cross-domain similarity (source -> target)
    cross_sim = torch.matmul(source_features, target_features.T)  # [n_s, n_t]
    
    # Pre kazdy source najdeme najblizsiu target vzorku
    soft_assignments = F.softmax(cross_sim / temperature, dim=1)
    
    # Loss - chceme maximalizovat cross-domain podobnost
    loss = -torch.mean(torch.log(soft_assignments.max(dim=1)[0] + 1e-6))
    
    return loss


class ContrastiveTrainer:
    """
    Contrastive Domain Alignment Trener.
    
    Celkova loss = classification_loss + lambda * contrastive_loss
    """

    def __init__(self, feature_extractor, label_classifier, device="cpu"):
        """
        Parametre:
            feature_extractor: siet na extrakciu features
            label_classifier: siet na klasifikaciu
            device: "cpu" alebo "cuda"
        """
        self.device = device
        
        self.feature_extractor = feature_extractor.to(device)
        self.label_classifier = label_classifier.to(device)
        
        # Projekcna hlavicka pre contrastive learning
        feature_dim = feature_extractor.get_feature_dim()
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            projection_dim=CONTRASTIVE_CONFIG["projection_dim"],
        ).to(device)
        
        # Loss funkcie
        self.label_criterion = nn.CrossEntropyLoss()
        self.temperature = CONTRASTIVE_CONFIG["temperature"]
        
        # Optimizer
        all_params = (
            list(self.feature_extractor.parameters()) +
            list(self.label_classifier.parameters()) +
            list(self.projection_head.parameters())
        )
        
        self.optimizer = optim.Adam(
            all_params,
            lr=CONTRASTIVE_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        
        # Historia
        self.train_losses = []
        self.contrastive_losses = []
        self.val_accuracies = []

    def train_epoch(self, source_loader, target_loader):
        """Jedna trenovacia epocha s contrastive alignmentom."""
        self.feature_extractor.train()
        self.label_classifier.train()
        self.projection_head.train()
        
        total_label_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0
        
        target_iter = iter(target_loader)
        
        for source_data, source_labels in source_loader:
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)
            
            # ===== DOPREDNY PRIECHOD =====
            
            # Extrahujeme features
            source_features = self.feature_extractor(source_data)
            target_features = self.feature_extractor(target_data)
            
            # 1. Label klasifikacia na source datach
            label_output = self.label_classifier(source_features)
            label_loss = self.label_criterion(label_output, source_labels)
            
            # 2. Projekcia pre contrastive learning
            source_projected = self.projection_head(source_features)
            target_projected = self.projection_head(target_features)
            
            # 3. Supervised contrastive loss na source datach
            sup_contrastive = supervised_contrastive_loss(
                source_projected, source_labels, self.temperature
            )
            
            # 4. Domain contrastive alignment
            domain_contrastive = domain_contrastive_loss(
                source_projected, target_projected, self.temperature
            )
            
            # Celkova contrastive loss
            contrastive_loss = sup_contrastive + domain_contrastive
            
            # Celkova loss
            total_loss = label_loss + CONTRASTIVE_CONFIG["lambda_contrastive"] * contrastive_loss
            
            # ===== SPATNY PRIECHOD =====
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_label_loss += label_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            num_batches += 1
        
        avg_label_loss = total_label_loss / max(num_batches, 1)
        avg_contrastive_loss = total_contrastive_loss / max(num_batches, 1)
        
        return avg_label_loss, avg_contrastive_loss

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

    def train(self, source_loader, target_loader, val_loader=None, num_epochs=None):
        """Kompletný contrastive trenovaci cyklus."""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n=== Contrastive Alignment trenovanie ({num_epochs} epoch) ===")
        
        for epoch in range(num_epochs):
            label_loss, contrastive_loss = self.train_epoch(source_loader, target_loader)
            self.train_losses.append(label_loss)
            self.contrastive_losses.append(contrastive_loss)
            
            val_info = ""
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_acc = val_results["accuracy"]
                self.val_accuracies.append(val_acc)
                val_info = f", Val Acc: {val_acc:.4f}"
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                    print(f"  Early stopping na epoche {epoch+1}")
                    break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epocha {epoch+1}/{num_epochs}: "
                      f"Label Loss = {label_loss:.4f}, "
                      f"Contrastive = {contrastive_loss:.4f}{val_info}")
        
        return {
            "train_losses": self.train_losses,
            "contrastive_losses": self.contrastive_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_val_acc,
        }
