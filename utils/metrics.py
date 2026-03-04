"""
Modul na pocitanie klasifikacnych metrik.
Tu su vsetky metriky ktore pouzivame na vyhodnotenie modelov:
- Accuracy (presnost)
- AUC (Area Under ROC Curve)
- F1-score (harmonicky priemer precision a recall)
- Senzitivita (recall pre PD triedu - kolko PD pacientov sme spravne odhalili)
- Specificita (recall pre Healthy triedu - kolko zdravych sme spravne identifikovali)

Pre klinicky system je dolezita hlavne senzitivita
(nechceme prehliadnut PD pacienta).

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    roc_curve,
)


def compute_accuracy(y_true, y_pred):
    """
    Spocita presnost (accuracy).
    Accuracy = pocet spravnych predikcii / celkovy pocet
    
    Parametre:
        y_true: skutocne labely
        y_pred: predikovane labely
    
    Vrati:
        float: accuracy (0-1)
    """
    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred, average="binary"):
    """
    Spocita F1-score.
    F1 = 2 * (precision * recall) / (precision + recall)
    
    F1 je dobra metrika ked mame nebalanciu tried
    (napr. viac healthy nez PD).
    
    Parametre:
        y_true: skutocne labely
        y_pred: predikovane labely
        average: typ priemerovania ("binary" pre dve triedy)
    
    Vrati:
        float: F1-score (0-1)
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_auc(y_true, y_prob):
    """
    Spocita AUC (Area Under ROC Curve).
    AUC meria kvalitu modelu nezavisle od thresholdu.
    AUC = 0.5 je nahodny model, AUC = 1.0 je perfektny model.
    
    Parametre:
        y_true: skutocne labely
        y_prob: pravdepodobnosti (pre pozitivnu triedu)
    
    Vrati:
        float: AUC hodnota (0-1)
    """
    try:
        # Ak mame pravdepodobnosti pre obe triedy, vezmeme druhu (PD)
        if y_prob.ndim == 2:
            y_prob_positive = y_prob[:, 1]
        else:
            y_prob_positive = y_prob
        
        return roc_auc_score(y_true, y_prob_positive)
    except ValueError:
        # Ak su vsetky labely rovnake, AUC sa neda vypocitat
        print("  VAROVANIE: AUC sa neda vypocitat (len jedna trieda v datach)")
        return 0.5


def compute_sensitivity(y_true, y_pred):
    """
    Spocita senzitivitu (sensitivity / recall pre PD triedu).
    Senzitivita = TP / (TP + FN)
    = kolko percent PD pacientov sme spravne identifikovali.
    
    V klinickej praxi je to najdolezitejsia metrika!
    Nechceme prehliadnut PD pacienta.
    
    Parametre:
        y_true: skutocne labely
        y_pred: predikovane labely
    
    Vrati:
        float: senzitivita (0-1)
    """
    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Ak nemame ziadne PD vzorky
    if cm.shape[0] < 2:
        return 0.0
    
    tn, fp, fn, tp = cm.ravel()
    
    # Senzitivita = TP / (TP + FN)
    if tp + fn == 0:
        return 0.0
    
    sensitivity = tp / (tp + fn)
    return sensitivity


def compute_specificity(y_true, y_pred):
    """
    Spocita specificitu (specificity / recall pre Healthy triedu).
    Specificita = TN / (TN + FP)
    = kolko percent zdravych sme spravne identifikovali ako zdravych.
    
    Parametre:
        y_true: skutocne labely
        y_pred: predikovane labely
    
    Vrati:
        float: specificita (0-1)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    if cm.shape[0] < 2:
        return 0.0
    
    tn, fp, fn, tp = cm.ravel()
    
    # Specificita = TN / (TN + FP)
    if tn + fp == 0:
        return 0.0
    
    specificity = tn / (tn + fp)
    return specificity


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Spocita VSETKY metriky naraz.
    Toto je hlavna funkcia ktoru volame pri evaluacii.
    
    Parametre:
        y_true: skutocne labely
        y_pred: predikovane labely
        y_prob: pravdepodobnosti (optional, pre AUC)
    
    Vrati:
        dict: slovnik so vsetkymi metrikami
    """
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_score": compute_f1(y_true, y_pred),
        "sensitivity": compute_sensitivity(y_true, y_pred),
        "specificity": compute_specificity(y_true, y_pred),
    }
    
    # AUC potrebuje pravdepodobnosti
    if y_prob is not None:
        metrics["auc"] = compute_auc(y_true, y_prob)
    else:
        metrics["auc"] = 0.5  # ak nemame prob, nastavime na nahodne
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm
    
    # Precision
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    
    # ROC krivka (pre vizualizaciu)
    if y_prob is not None:
        if y_prob.ndim == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        except ValueError:
            metrics["roc_curve"] = None
    
    return metrics
