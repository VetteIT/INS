"""
Baseline pristup - BEZ domain adaptation.
Jednoducho natrénujeme model na source domene a otestujeme na target domene.
Toto slúži ako zakladna linia (baseline) s ktorou porovnavame
vsetky domain adaptation techniky.

Ocakavame ze baseline bude mat horsie vysledky nez DA techniky,
pretoze model nevidel data z target domeny pri trenovani.

Autori: Dmytro Protsun, Mykyta Olym
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config.settings import TRAINING_CONFIG


class BaselineTrainer:
    """
    Baseline trener - trénuje model bez akejkolvek domain adaptation.
    
    Postup:
    1. Natrénujeme model na source domene (kde mame labely)
    2. Otestujeme na target domene (ina domena, iny dataset)
    3. Pozrieme sa kolko sa zhorsil vykon
    """

    def __init__(self, model, device="cpu"):
        """
        Parametre:
            model: PyTorch model (CNN, MLP, Wav2Vec...)
            device: zariadenie ("cpu" alebo "cuda")
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss funkcia - CrossEntropy pretoze mame 2 triedy
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - Adam je najrozsirenejsi
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG.get("learning_rate", 0.001),
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )
        
        # Na ukladanie straty pocas trenovania
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader):
        """
        Natrénuje model na jednej epoche.
        
        Parametre:
            train_loader: DataLoader so source datami
        
        Vrati:
            float: priemerná strata (loss)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, batch_labels in train_loader:
            # Presunieme data na spravne zariadenie (CPU/GPU)
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Vynulujeme gradienty
            self.optimizer.zero_grad()
            
            # Dopredny priechod
            outputs = self.model(batch_data)
            
            # Spocitame stratu
            loss = self.criterion(outputs, batch_labels)
            
            # Spatny priechod (backpropagation)
            loss.backward()
            
            # Aktualizujeme vahy
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Priemerna strata za epochu
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, test_loader):
        """
        Vyhodnoti model na testovacich datach.
        
        Parametre:
            test_loader: DataLoader s testovacimi datami
        
        Vrati:
            dict: slovnik s metrikami (accuracy, predictions, labels)
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():  # nepotrebujeme gradienty pri evaluacii
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Predikcia
                outputs = self.model(batch_data)
                
                # Pravdepodobnosti (softmax)
                probs = torch.softmax(outputs, dim=1)
                
                # Predikovane triedy (argmax)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Spocitame accuracy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = np.mean(all_predictions == all_labels)
        
        results = {
            "accuracy": accuracy,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probs,
        }
        
        return results

    def train(self, train_loader, val_loader=None, num_epochs=None):
        """
        Kompletny trenovaci cyklus.
        
        Parametre:
            train_loader: DataLoader s trenovacimi datami
            val_loader: DataLoader s validacnymi datami (optional)
            num_epochs: pocet epoch
        
        Vrati:
            dict: slovnik s historiou trenovania
        """
        if num_epochs is None:
            num_epochs = TRAINING_CONFIG["num_epochs"]
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"\n=== Baseline trenovanie ({num_epochs} epoch) ===")
        
        for epoch in range(num_epochs):
            # Trenovanie
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validacia (ak mame validacne data)
            val_info = ""
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_acc = val_results["accuracy"]
                self.val_accuracies.append(val_acc)
                val_info = f", Val Acc: {val_acc:.4f}"
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                
                if patience_counter >= TRAINING_CONFIG["early_stopping_patience"]:
                    print(f"  Early stopping na epoche {epoch+1}")
                    break
            
            # Vypis kazdych 5 epoch
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epocha {epoch+1}/{num_epochs}: Loss = {train_loss:.4f}{val_info}")
        
        # Nacitame najlepsi model ak mame
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"  Najlepsia validacna presnost: {best_val_acc:.4f}")
        
        history = {
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": best_val_acc,
        }
        
        return history
