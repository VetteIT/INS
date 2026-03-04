"""
DANN - Domain-Adversarial Neural Network.
Toto je jedna z najznamejsich domain adaptation technik.

Princip DANN:
    Model sa sklada z troch casti:
    1. Feature Extractor (G) - extrahuje features z vstupu
    2. Label Classifier (C) - klasifikuje PD vs Healthy
    3. Domain Classifier (D) - klasifikuje z ktorej domeny data pochadza

    Finta je v tom ze pouzivame GRADIENT REVERSAL LAYER medzi
    feature extractorom a domain classifierom. To znamena ze:
    - Label classifier sa snazi spravne klasifikovat PD/Healthy
    - Domain classifier sa snazi urcit domenu
    - Ale gradient reversal sposobi ze feature extractor sa snazi
      ZMIAST domain classifier (features su domain-invariantne)
    
    Vysledok: Features ktore su uzitocne pre klasifikaciu PD,
              ale nezavisia na domene (datasete).

Paper: "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)

Autori: Dmytro Protsun, Mykyta Olym
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Function

from config.settings import DANN_CONFIG, TRAINING_CONFIG

# Dropout pre domain classifier - berieme z DANN configu
DANN_DROPOUT = DANN_CONFIG.get("dropout_rate", 0.3)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL).
    Pri doprednom priechode nerobia nic (identity).
    Pri spatnom priechode nasobena gradient -lambda (obracena ho).
    
    Toto je klucova cast DANN - GRL sposobi ze feature extractor
    sa uci features ktore ZMATU domain classifier.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_val):
        """Dopredny priechod - nic nerobime, len ulozime lambda."""
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Spatny priechod - obracime gradient (nasobime -lambda)."""
        output = grad_output.neg() * ctx.lambda_val
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Wrapper okolo GradientReversalFunction aby sa lahsie pouzival.
    """
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val
    
    def set_lambda(self, lambda_val):
        """Nastavi lambda parameter."""
        self.lambda_val = lambda_val
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DomainClassifier(nn.Module):
    """
    Domain Classifier - snazi sa urcit z ktorej domeny data pochadza.
    Je to jednoducha siet s 2-3 vrstvami.
    """
    
    def __init__(self, feature_dim, hidden_dim=256, num_domains=2):
        """
        Parametre:
            feature_dim: velkost vstupneho feature vektora
            hidden_dim: velkost skrytej vrstvy
            num_domains: pocet domen (2 pre source/target)
        """
        super(DomainClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DANN_DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DANN_DROPOUT),
            nn.Linear(hidden_dim // 2, num_domains),
        )
    
    def forward(self, x):
        return self.classifier(x)


class DANNTrainer:
    """
    DANN Trener - implementuje trenovanie s domain-adversarial pristupom.
    
    Trenujeme tri casti sucasne:
    1. Feature extractor - generuje domain-invariantne features
    2. Label classifier - klasifikuje PD/Healthy (len na source datach)
    3. Domain classifier - rozlisuje source/target (na oboch)
    """

    def __init__(self, feature_extractor, label_classifier, device="cpu"):
        """
        Parametre:
            feature_extractor: siet na extrakciu features
            label_classifier: siet na klasifikaciu PD/Healthy
            device: "cpu" alebo "cuda"
        """
        self.device = device
        
        # Hlavne casti modelu
        self.feature_extractor = feature_extractor.to(device)
        self.label_classifier = label_classifier.to(device)
        
        # Zistíme velkost feature vektora
        feature_dim = feature_extractor.get_feature_dim()
        
        # Domain classifier
        self.domain_classifier = DomainClassifier(
            feature_dim=feature_dim,
            hidden_dim=DANN_CONFIG["hidden_dim"],
            num_domains=2,  # source vs target
        ).to(device)
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_val=DANN_CONFIG["lambda_domain"])
        
        # Loss funkcie
        self.label_criterion = nn.CrossEntropyLoss()    # pre PD/Healthy
        self.domain_criterion = nn.CrossEntropyLoss()   # pre source/target
        
        # Optimizer pre vsetky casti spolu
        all_params = (
            list(self.feature_extractor.parameters()) +
            list(self.label_classifier.parameters()) +
            list(self.domain_classifier.parameters())
        )
        
        self.optimizer = optim.Adam(
            all_params,
            lr=DANN_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        
        # Historia
        self.train_losses = []
        self.domain_losses = []
        self.val_accuracies = []

    def _compute_lambda(self, epoch, num_epochs):
        """
        Vypocita lambda parameter pre GRL.
        Lambda sa postupne zvysuje pocas trenovania (schedule).
        Na zaciatku je mala (model sa najprv uci klasifikovat),
        potom sa zvysuje (zaciname "pliest" domain classifier).
        
        Pouzivame formula z originalneho DANN paperu:
        lambda = 2 / (1 + exp(-alpha * p)) - 1
        kde p = epoch / num_epochs
        """
        p = epoch / num_epochs
        lambda_val = 2.0 / (1.0 + np.exp(-DANN_CONFIG["alpha"] * p)) - 1
        return lambda_val

    def train_epoch(self, source_loader, target_loader, epoch, num_epochs):
        """
        Natrénuje model na jednej epoche s DANN.
        
        Parametre:
            source_loader: DataLoader so source datami (s labelmi)
            target_loader: DataLoader s target datami (bez labelov)
            epoch: cislo aktualnej epochy
            num_epochs: celkovy pocet epoch
        
        Vrati:
            tuple: (label_loss, domain_loss)
        """
        self.feature_extractor.train()
        self.label_classifier.train()
        self.domain_classifier.train()
        
        # Aktualizujeme lambda podla schedule
        lambda_val = self._compute_lambda(epoch, num_epochs)
        self.grl.set_lambda(lambda_val)
        
        total_label_loss = 0.0
        total_domain_loss = 0.0
        num_batches = 0
        
        # Iterujeme cez source aj target data sucasne
        target_iter = iter(target_loader)
        
        for source_data, source_labels in source_loader:
            # Ziskame target batch (ak dosli, zaciname od znova)
            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data, _ = next(target_iter)
            
            # Presunieme na device
            source_data = source_data.to(self.device)
            source_labels = source_labels.to(self.device)
            target_data = target_data.to(self.device)
            
            # Velkost batchov
            batch_size_source = source_data.size(0)
            batch_size_target = target_data.size(0)
            
            # Domain labely (0 = source, 1 = target)
            source_domain_labels = torch.zeros(batch_size_source, dtype=torch.long).to(self.device)
            target_domain_labels = torch.ones(batch_size_target, dtype=torch.long).to(self.device)
            
            # ===== DOPREDNY PRIECHOD =====
            
            # 1. Extrahujeme features zo source dat
            source_features = self.feature_extractor(source_data)
            
            # 2. Label klasifikacia (len na source datach kde mame labely)
            label_output = self.label_classifier(source_features)
            label_loss = self.label_criterion(label_output, source_labels)
            
            # 3. Domain klasifikacia na source datach (s GRL!)
            source_domain_input = self.grl(source_features)
            source_domain_output = self.domain_classifier(source_domain_input)
            source_domain_loss = self.domain_criterion(source_domain_output, source_domain_labels)
            
            # 4. Extrahujeme features z target dat
            target_features = self.feature_extractor(target_data)
            
            # 5. Domain klasifikacia na target datach (s GRL!)
            target_domain_input = self.grl(target_features)
            target_domain_output = self.domain_classifier(target_domain_input)
            target_domain_loss = self.domain_criterion(target_domain_output, target_domain_labels)
            
            # Celkova domain loss
            domain_loss = (source_domain_loss + target_domain_loss) / 2
            
            # Celkova loss
            total_loss = label_loss + DANN_CONFIG["lambda_domain"] * domain_loss
            
            # ===== SPATNY PRIECHOD =====
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            num_batches += 1
        
        avg_label_loss = total_label_loss / max(num_batches, 1)
        avg_domain_loss = total_domain_loss / max(num_batches, 1)
        
        return avg_label_loss, avg_domain_loss

    def evaluate(self, test_loader):
        """
        Vyhodnoti model na testovacich datach.
        
        Parametre:
            test_loader: DataLoader s testovacimi datami
        
        Vrati:
            dict: vysledky evaluacie
        """
        self.feature_extractor.eval()
        self.label_classifier.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Extrahujeme features
                features = self.feature_extractor(batch_data)
                
                # Klasifikacia
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
        """
        Kompletny DANN trenovaci cyklus.
        """
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n=== DANN trenovanie ({num_epochs} epoch) ===")
        
        for epoch in range(num_epochs):
            # Trenovanie
            label_loss, domain_loss = self.train_epoch(
                source_loader, target_loader, epoch, num_epochs
            )
            self.train_losses.append(label_loss)
            self.domain_losses.append(domain_loss)
            
            # Lambda info
            lambda_val = self._compute_lambda(epoch, num_epochs)
            
            # Validacia
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
                        'domain_classifier': copy.deepcopy(self.domain_classifier.state_dict()),
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                    print(f"  Early stopping na epoche {epoch+1}")
                    break
            
            # Vypis
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epocha {epoch+1}/{num_epochs}: "
                      f"Label Loss = {label_loss:.4f}, "
                      f"Domain Loss = {domain_loss:.4f}, "
                      f"λ = {lambda_val:.4f}{val_info}")
        
        # Nacitame najlepsi model ak mame
        if best_model_state is not None:
            self.feature_extractor.load_state_dict(best_model_state['feature_extractor'])
            self.label_classifier.load_state_dict(best_model_state['label_classifier'])
            self.domain_classifier.load_state_dict(best_model_state['domain_classifier'])
            print(f"  Najlepsia validacna presnost: {best_val_acc:.4f}")
        
        history = {
            "train_losses": self.train_losses,
            "domain_losses": self.domain_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_val_acc,
        }
        
        return history
