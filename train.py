"""
Týždeň 4-8: Trénovacie a evaluačné funkcie.

Trénovací cyklus: Cvičenie 4 - FF.ipynb
  for epoch → for batch → forward → loss → backward → step

Evaluácia: Cvičenie 4 - with torch.no_grad(): ...
"""

import torch
import torch.nn as nn
from itertools import cycle

from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MMD_LAMBDA, GRAD_CLIP_NORM, WEIGHT_DECAY


# ========================================================================
# Trénovanie (Týždeň 4-5)
# Vzor: Cvičenie 4 - FF.ipynb, trénovací cyklus
# ========================================================================

def train_model(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                class_weights=None):
    """
    Štandardné trénovanie klasifikátora (bez domain adaptácie).
    Vzor: Cvičenie 4 - trénovací cyklus na MNIST

    Args:
        model: PyTorch model (MLP alebo CNN1D)
        train_loader: DataLoader s trénovacími dátami
        num_epochs: počet epoch
        lr: learning rate
        class_weights: váhy tried pre vyváženú loss (voliteľné)
    """
    model = model.to(DEVICE)
    model.train()

    # Loss a optimalizátor - vzor z Cvičenie 4
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for features, labels in train_loader:
            # Presun na zariadenie - vzor z Cvičenie 4
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass - vzor z Cvičenie 4
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Výpis každých 10 epoch
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return losses


# ========================================================================
# Trénovanie DANN (Týždeň 7)
# Ref: Ganin et al. (2015), Algorithm 1
# https://arxiv.org/abs/1409.7495
# ========================================================================

def train_dann(model, source_loader, target_loader,
               num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, class_weights=None):
    """
    Trénovanie DANN - adversariálna domain adaptácia.

    Source loader: dáta s labelmi (PD/Healthy) zo zdrojovej domény
    Target loader: dáta BEZ labelov z cieľovej domény
                   (labely sa NEPOUŽÍVAJÚ pri trénovaní)
    """
    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    domain_criterion = nn.CrossEntropyLoss()  # doménová strata bez váh
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Uložíme pôvodnú max lambda pre progresívne zvyšovanie
    max_lambda = model.grl.lambda_val

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Progresívna lambda: lineárne zvyšovanie od 0 do max_lambda
        # Pôvodný paper: λ_p = 2/(1+exp(-γ·p))-1, my použijeme lineárnu rampu
        # pre jednoduchosť a stabilitu trénovania
        progress = epoch / max(num_epochs - 1, 1)
        current_lambda = max_lambda * progress
        model.grl.lambda_val = current_lambda

        # zip + cycle: iterujeme cez oba loadery súčasne
        # cycle opakuje kratší loader
        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            # Forward pass
            src_class, src_domain, _ = model(src_x)
            _, tgt_domain, _ = model(tgt_x)

            # Klasifikačná strata (len na source!)
            class_loss = criterion(src_class, src_y)

            # Doménová strata (na oboch doménach)
            # Source = 0, Target = 1
            src_domain_labels = torch.zeros(src_x.size(0), dtype=torch.long,
                                            device=DEVICE)
            tgt_domain_labels = torch.ones(tgt_x.size(0), dtype=torch.long,
                                           device=DEVICE)
            domain_loss = (domain_criterion(src_domain, src_domain_labels) +
                           domain_criterion(tgt_domain, tgt_domain_labels))

            # Celková strata
            # GRL v modeli sa stará o obrátenie gradientov domain_loss
            loss = class_loss + domain_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(λ={current_lambda:.3f}, '
                  f'class: {class_loss.item():.4f}, '
                  f'domain: {domain_loss.item():.4f})')

    return losses


# ========================================================================
# Trénovanie s MMD (Týždeň 8)
# Ref: Gretton et al. (2012)
# ========================================================================

def train_mmd(model, source_loader, target_loader,
              num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, mmd_lambda=MMD_LAMBDA,
              class_weights=None):
    """
    Trénovanie s MMD stratou.
    Klasifikačná strata + MMD strata medzi zdrojovými a cieľovými príznakmi.
    """
    from domain_adaptation import mmd_loss

    model = model.to(DEVICE)
    model.train()

    weight = class_weights.to(DEVICE) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for (src_x, src_y), (tgt_x, _) in zip(source_loader, cycle(target_loader)):
            src_x = src_x.to(DEVICE)
            src_y = src_y.to(DEVICE)
            tgt_x = tgt_x.to(DEVICE)

            # Forward pass
            src_output, src_features = model(src_x)
            _, tgt_features = model(tgt_x)

            # Klasifikačná strata
            class_loss = criterion(src_output, src_y)

            # MMD strata - zarovnanie distribúcií
            mmd = mmd_loss(src_features, tgt_features)

            # Celková strata
            loss = class_loss + mmd_lambda * mmd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f} '
                  f'(class: {class_loss.item():.4f}, '
                  f'MMD: {mmd.item():.4f})')

    return losses
