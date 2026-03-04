"""
Modul na vizualizaciu vysledkov.
Tu su funkcie na kreslenie grafov:
- Confusion matrix (matica zamien)
- ROC krivka
- Training history (loss a accuracy v case)
- Porovnanie DA technik
- t-SNE vizualizacia feature priestoru

Pouzivame matplotlib na kreslenie.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend bez GUI (aby fungovalo na serveri)

from config.settings import RESULTS_DIR


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix",
                          save_path=None):
    """
    Nakresli confusion matrix (maticu zamien).
    Ukazuje kolko vzoriek bolo spravne/nespravne klasifikovanych.
    
    Parametre:
        cm (numpy array): confusion matrix [2x2]
        class_names (list): nazvy tried
        title (str): nadpis grafu
        save_path (str): cesta na ulozenie
    """
    if class_names is None:
        class_names = ["Healthy", "PD"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Nakreslime farebnu maticu
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Nastavime osi
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Skutocny label",
        xlabel="Predikovany label",
        title=title,
    )
    
    # Zapiseme cisla do buniek
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graf ulozeny: {save_path}")
    
    plt.close()


def plot_roc_curve(roc_data, auc_value, title="ROC Curve", save_path=None):
    """
    Nakresli ROC krivku.
    ROC krivka ukazuje vztah medzi senzitivitou a (1-specificitou)
    pre rozne prahy klasifikacie.
    
    Parametre:
        roc_data (dict): slovnik s fpr, tpr, thresholds
        auc_value (float): AUC hodnota
        title: nadpis
        save_path: cesta na ulozenie
    """
    if roc_data is None:
        print("  Nemoze nakreslit ROC - chybaju data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ROC krivka
    ax.plot(roc_data["fpr"], roc_data["tpr"],
            color="darkorange", lw=2,
            label=f"ROC (AUC = {auc_value:.4f})")
    
    # Diagonala (nahodny model)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
            label="Nahodny model (AUC = 0.5)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificita)")
    ax.set_ylabel("True Positive Rate (Senzitivita)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graf ulozeny: {save_path}")
    
    plt.close()


def plot_training_history(history, title="Training History", save_path=None):
    """
    Nakresli historiu trenovania - loss a accuracy cez epochy.
    Pomaha nam vidiet ci model konverguje a nema overfitting.
    
    Parametre:
        history (dict): slovnik s train_losses, val_accuracies...
        title: nadpis
        save_path: cesta na ulozenie
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graf 1: Loss
    if "train_losses" in history:
        axes[0].plot(history["train_losses"], label="Train Loss")
    if "domain_losses" in history:
        axes[0].plot(history["domain_losses"], label="Domain Loss", linestyle="--")
    if "mmd_losses" in history:
        axes[0].plot(history["mmd_losses"], label="MMD Loss", linestyle="--")
    if "contrastive_losses" in history:
        axes[0].plot(history["contrastive_losses"], label="Contrastive Loss", linestyle="--")
    
    axes[0].set_xlabel("Epocha")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Graf 2: Accuracy
    if "val_accuracies" in history and len(history["val_accuracies"]) > 0:
        axes[1].plot(history["val_accuracies"], label="Val Accuracy", color="green")
        axes[1].set_xlabel("Epocha")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Ziadne validacne data",
                     ha="center", va="center", fontsize=12)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graf ulozeny: {save_path}")
    
    plt.close()


def plot_da_comparison(results_dict, metric_name="f1_score",
                       title="Porovnanie DA technik", save_path=None):
    """
    Nakresli stlpcovy graf porovnavajuci viacero DA technik.
    
    Parametre:
        results_dict (dict): {nazov_metody: {metriky...}}
        metric_name: ktoru metriku chceme porovnat
        title: nadpis
        save_path: cesta na ulozenie
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results_dict.keys())
    values = [results_dict[m].get(metric_name, 0) for m in methods]
    
    # Farby pre rozne metody
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    bars = ax.bar(methods, values, color=bar_colors, edgecolor="black", alpha=0.8)
    
    # Pridame hodnoty nad stlpce
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    
    # Otocime labely ak su dlhe
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Graf ulozeny: {save_path}")
    
    plt.close()
