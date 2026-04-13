"""
Týždeň 7-8: Domain Adaptation techniky.

Implementácia:
  1. Gradient Reversal Layer (GRL) - kľúčový komponent DANN
  2. DANN model - Domain-Adversarial Neural Network
  3. MMD loss - Maximum Mean Discrepancy

Tieto techniky pomáhajú modelu naučiť sa príznaky, ktoré sú
nezávislé od domény (nemocnice/datasetu), ale stále užitočné
pre klasifikáciu PD vs Healthy.

Ref: Ganin, Y. & Lempitsky, V. (2015)
     "Unsupervised Domain Adaptation by Backpropagation"
     https://arxiv.org/abs/1409.7495

Ref: Gretton, A. et al. (2012)
     "A Kernel Two-Sample Test"
     https://jmlr.org/papers/v13/gretton12a.html
"""

import torch
import torch.nn as nn
from config import (NUM_FEATURES, NUM_CLASSES, NUM_DOMAINS,
                    HIDDEN_SIZE, FEATURE_DIM, DANN_LAMBDA, DROPOUT_RATE)


# ========================================================================
# Gradient Reversal Layer (GRL)
# Zdroj: Ganin et al. (2015), https://arxiv.org/abs/1409.7495
#
# Pri forward pass: identita (ničí nerobí)
# Pri backward pass: obráti smer gradientov (* -lambda)
#
# Toto núti feature extractor vytvárať príznaky, z ktorých
# sa NEDÁ určiť doména → doménovo-invariantné príznaky
# ========================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Vlastná autograd funkcia pre reverziu gradientov.
    https://pytorch.org/docs/stable/autograd.html#function
    """

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Kľúčová operácia: obrátenie gradientov
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL modul - obalenie funkcie do nn.Module."""

    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


# ========================================================================
# DANN - Domain-Adversarial Neural Network
# Vzor: Cvičenie 3-4 (nn.Module architektúra, forward pass)
#
# Architektúra:
#   Feature Extractor → [Label Predictor]
#                    ↘ [GRL → Domain Classifier]
#
# Feature Extractor: naučí sa doménovo-invariantné príznaky
# Label Predictor: klasifikuje PD vs Healthy (len na zdrojovej doméne)
# Domain Classifier: rozlišuje domény (GRL obracia gradienty)
# ========================================================================

class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network.
    Ref: Ganin et al. (2015), https://arxiv.org/abs/1409.7495
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 num_domains=NUM_DOMAINS, lambda_val=DANN_LAMBDA,
                 dropout=DROPOUT_RATE):
        super(DANNModel, self).__init__()

        # Feature Extractor (zdieľaný) - vzor z Cvičenie 3-4
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU()
        )

        # Label Predictor - klasifikácia PD vs Healthy
        self.label_predictor = nn.Linear(feature_dim, num_classes)

        # Domain Classifier s GRL - rozlišovanie domén
        self.grl = GradientReversalLayer(lambda_val)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_domains)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output, features


# ========================================================================
# MMD Loss - Maximum Mean Discrepancy
# Ref: Gretton et al. (2012)
#
# Meria vzdialenosť medzi distribúciami príznakov dvoch domén.
# Ak je MMD malé → distribúcie sú podobné → doménová adaptácia funguje.
#
# Jednoduchá lineárna MMD (priemer príznakov):
#   MMD² ≈ ||mean(source) - mean(target)||²
# ========================================================================

def mmd_loss(source_features, target_features):
    """
    Lineárna MMD strata medzi zdrojovými a cieľovými príznakmi.

    Args:
        source_features: príznaky zo zdrojovej domény (batch, feature_dim)
        target_features: príznaky z cieľovej domény (batch, feature_dim)

    Returns:
        MMD² hodnota (scalar)
    """
    mean_source = source_features.mean(dim=0)
    mean_target = target_features.mean(dim=0)
    return torch.sum((mean_source - mean_target) ** 2)


# ========================================================================
# MMD Model - klasifikátor s MMD stratou
# Rovnaký feature extractor, ale namiesto GRL pridávame MMD loss
# do celkovej straty pri trénovaní.
# ========================================================================

class MMDModel(nn.Module):
    """
    Klasifikátor s MMD domain adaptation.
    Feature extractor sa učí mapovať obe domény na podobné distribúcie.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE):
        super(MMDModel, self).__init__()

        # Rovnaký feature extractor ako v DANN
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features



