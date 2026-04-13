"""
Týždeň 6: Evaluácia modelov.
Vzor: Cvičenie 4 - FF.ipynb (evaluácia s torch.no_grad)

Metriky pre klinické nasadenie:
  - AUC (Area Under ROC Curve)
  - F1 skóre
  - Senzitivita (recall) - schopnosť zachytiť PD pacientov
  - Špecificita - schopnosť správne identifikovať zdravých

Ref: sklearn metriky
https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.config import DEVICE


def evaluate_model(model, test_loader, model_type='standard'):
    """
    Evaluácia modelu na testovacej sade.
    Vzor: Cvičenie 4 - with torch.no_grad(): ...

    Args:
        model: natrénovaný PyTorch model
        test_loader: DataLoader s testovacími dátami
        model_type: 'standard' (MLP/CNN) alebo 'dann' alebo 'mmd'

    Returns:
        dict s metrikami (accuracy, f1, auc, sensitivity, specificity)
    """
    model = model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # Vzor z Cvičenia 4: evaluácia bez gradientov
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)

            # Forward pass podľa typu modelu
            if model_type == 'dann':
                outputs, _, _ = model(features)
            elif model_type == 'mmd':
                outputs, _ = model(features)
            else:
                outputs = model(features)

            # Pravdepodobnosti a predikcie
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Výpočet metrík
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC - vyžaduje obe triedy v dátach
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Senzitivita a špecificita z konfúznej matice
    if len(np.unique(all_labels)) == 2:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def evaluate_svm(svm_model, X_test, y_test):
    """
    Evaluácia SVM modelu (scikit-learn).

    Args:
        svm_model: natrénovaný Pipeline (scaler + SVM)
        X_test: testovacie príznaky
        y_test: testovacie labely
    """
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    return {
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def print_metrics(name, metrics):
    """Prehľadný výpis metrík."""
    print(f"  {name}:")
    print(f"    Accuracy:    {metrics['accuracy']:.3f}")
    print(f"    F1:          {metrics['f1']:.3f}")
    print(f"    AUC:         {metrics['auc']:.3f}")
    print(f"    Senzitivita: {metrics['sensitivity']:.3f}")
    print(f"    Špecificita: {metrics['specificity']:.3f}")


def print_comparison_table(results):
    """
    Tabuľka porovnania výsledkov.

    Args:
        results: dict of dicts, napr. {'MLP in-domain': {...}, 'DANN': {...}}
    """
    print("\n" + "=" * 75)
    print("POROVNANIE VÝSLEDKOV")
    print("=" * 75)
    header = f"{'Metóda':<30} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Sens':>6} {'Spec':>6}"
    print(header)
    print("-" * 75)

    for name, m in results.items():
        row = (f"{name:<30} "
               f"{m['accuracy']:>6.3f} "
               f"{m['f1']:>6.3f} "
               f"{m['auc']:>6.3f} "
               f"{m['sensitivity']:>6.3f} "
               f"{m['specificity']:>6.3f}")
        print(row)

    print("=" * 75)
