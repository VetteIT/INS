"""
MMD (Maximum Mean Discrepancy) Domain Adaptation.
MMD meria vzdialenost medzi dvoma distribuciami vo feature priestore.

Princip:
    Chceme aby features z source a target domeny boli co najpodobnejsie.
    MMD je statisticky test ktory meria tuto podobnost.
    Minimalizaciou MMD zblizujeme distribúcie features oboch domen.

Formula:
    MMD^2 = E[k(xs, xs')] + E[k(xt, xt')] - 2 * E[k(xs, xt)]
    kde k() je kernel funkcia (typicky RBF/Gaussian)

Pouzivame multi-kernel MMD (MK-MMD) s viacerými sigami
pre lepsiu estimaciu.

Paper: "Learning Transferable Features with Deep Adaptation Networks" (Long et al., 2015)

Autori: Dmytro Protsun, Mykyta Olym
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config.settings import MMD_CONFIG, TRAINING_CONFIG


def compute_kernel(x, y, kernel_type="rbf", bandwidth=1.0):
    """
    Vypocita kernel maticu medzi dvoma sadami vzoriek.
    
    Parametre:
        x (torch.Tensor): vzorky z prvej distribúcie [n, d]
        y (torch.Tensor): vzorky z druhej distribúcie [m, d]
        kernel_type (str): typ kernelu ("rbf" alebo "linear")
        bandwidth (float): bandwidth parameter pre RBF kernel
    
    Vrati:
        torch.Tensor: kernel matica [n, m]
    """
    if kernel_type == "rbf":
        # RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        # Rozsirime dimenzie pre broadcasting
        x = x.unsqueeze(1)  # [n, 1, d]
        y = y.unsqueeze(0)  # [1, m, d]
        
        # Vypocitame ||x-y||^2
        diff = x - y
        dist = torch.sum(diff ** 2, dim=-1)  # [n, m]
        
        # Kernel hodnota
        kernel = torch.exp(-dist / (2.0 * bandwidth))
        
        return kernel
    
    elif kernel_type == "linear":
        # Linearny kernel: k(x,y) = x^T * y
        return torch.mm(x, y.t())
    
    else:
        raise ValueError(f"Neznamy kernel typ: {kernel_type}")


def compute_mmd(source_features, target_features, kernel_type="rbf", bandwidths=None):
    """
    Vypocita MMD (Maximum Mean Discrepancy) medzi source a target features.
    Pouzivame multi-kernel pristup s viacerymi bandwidths pre lepsiu estimaciu.
    
    Parametre:
        source_features (torch.Tensor): features z source domeny [n_s, d]
        target_features (torch.Tensor): features z target domeny [n_t, d]
        kernel_type: typ kernelu
        bandwidths: list bandwidth hodnot pre multi-kernel MMD
    
    Vrati:
        torch.Tensor: MMD hodnota (scalar)
    """
    if bandwidths is None:
        bandwidths = MMD_CONFIG["kernel_bandwidth"]
    
    mmd_value = 0.0
    
    # Multi-kernel MMD - secitame MMD pre rozne bandwidths
    for bandwidth in bandwidths:
        # Kernel medzi source-source
        k_ss = compute_kernel(source_features, source_features, kernel_type, bandwidth)
        
        # Kernel medzi target-target
        k_tt = compute_kernel(target_features, target_features, kernel_type, bandwidth)
        
        # Kernel medzi source-target
        k_st = compute_kernel(source_features, target_features, kernel_type, bandwidth)
        
        # MMD^2 = mean(k_ss) + mean(k_tt) - 2 * mean(k_st)
        mmd_sq = torch.mean(k_ss) + torch.mean(k_tt) - 2 * torch.mean(k_st)
        
        mmd_value += mmd_sq
    
    # Priemer cez bandwidths
    mmd_value = mmd_value / len(bandwidths)
    
    return mmd_value


class MMDTrainer:
    """
    MMD Domain Adaptation Trener.
    
    Celkova loss = classification loss + lambda * MMD loss
    
    Classification loss ucí model klasifikovat PD/Healthy na source datach.
    MMD loss zblizuje feature distribucie source a target domeny.
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
        
        # Loss funkcie
        self.label_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        all_params = (
            list(self.feature_extractor.parameters()) +
            list(self.label_classifier.parameters())
        )
        
        self.optimizer = optim.Adam(
            all_params,
            lr=MMD_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        
        # Historia
        self.train_losses = []
        self.mmd_losses = []
        self.val_accuracies = []

    def train_epoch(self, source_loader, target_loader):
        """
        Jedna trenovacia epocha s MMD regularizaciou.
        """
        self.feature_extractor.train()
        self.label_classifier.train()
        
        total_label_loss = 0.0
        total_mmd_loss = 0.0
        num_batches = 0
        
        target_iter = iter(target_loader)
        
        for source_data, source_labels in source_loader:
            # Target data
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
            
            # Label klasifikacia na source datach
            label_output = self.label_classifier(source_features)
            label_loss = self.label_criterion(label_output, source_labels)
            
            # MMD medzi source a target features
            mmd_loss = compute_mmd(source_features, target_features)
            
            # Celkova loss
            total_loss = label_loss + MMD_CONFIG["lambda_mmd"] * mmd_loss
            
            # ===== SPATNY PRIECHOD =====
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_label_loss += label_loss.item()
            total_mmd_loss += mmd_loss.item()
            num_batches += 1
        
        avg_label_loss = total_label_loss / max(num_batches, 1)
        avg_mmd_loss = total_mmd_loss / max(num_batches, 1)
        
        return avg_label_loss, avg_mmd_loss

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
        """Kompletny MMD trenovaci cyklus."""
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n=== MMD trenovanie ({num_epochs} epoch) ===")
        
        for epoch in range(num_epochs):
            label_loss, mmd_loss = self.train_epoch(source_loader, target_loader)
            self.train_losses.append(label_loss)
            self.mmd_losses.append(mmd_loss)
            
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
                      f"MMD Loss = {mmd_loss:.6f}{val_info}")
        
        # Nacitame najlepsi model ak mame
        if best_model_state is not None:
            self.feature_extractor.load_state_dict(best_model_state['feature_extractor'])
            self.label_classifier.load_state_dict(best_model_state['label_classifier'])
            print(f"  Najlepsia validacna presnost: {best_val_acc:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "mmd_losses": self.mmd_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_val_acc,
        }
