"""
Pomocné funkcie - vizualizácia a utility.
Vzor: Cvičenie 2 - matplotlib vizualizácia
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """Nastavenie seedu pre reprodukovateľnosť."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def plot_losses(losses_dict, title="Trénovacie straty", save_path=None):
    """
    Vykreslenie trénovacích strát.
    Vzor: matplotlib z Cvičenie 2

    Args:
        losses_dict: dict {'názov': [loss1, loss2, ...]}
        title: nadpis grafu
        save_path: cesta na uloženie (voliteľné)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ľavý graf: všetky modely
    for name, losses in losses_dict.items():
        axes[0].plot(losses, label=name)
    axes[0].set_xlabel('Epocha')
    axes[0].set_ylabel('Strata (Loss)')
    axes[0].set_title(f'{title} (všetky)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Pravý graf: priblíženie bez DA metód (iný rozsah strát)
    da_keywords = ('DANN', 'MMD')
    for name, losses in losses_dict.items():
        if not any(k in name for k in da_keywords):
            axes[1].plot(losses, label=name)
    axes[1].set_xlabel('Epocha')
    axes[1].set_ylabel('Strata (Loss)')
    axes[1].set_title(f'{title} (in-domain priblíženie)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.show()


def plot_comparison(results, save_path=None):
    """
    Bar chart porovnanie metrík pre rôzne metódy.
    Rozdelené na 3 skupiny: In-domain A, In-domain B, Cross-domain.

    Args:
        results: dict of dicts s metrikami
        save_path: cesta na uloženie (voliteľné)
    """
    metrics = ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity']
    labels = ['Accuracy', 'F1', 'AUC', 'Senzitivita', 'Špecificita']

    # Rozdelenie na skupiny pre prehľadnosť
    groups = {
        'In-domain A': {k: v for k, v in results.items() if 'in-domain A' in k},
        'In-domain B': {k: v for k, v in results.items() if 'in-domain B' in k},
        'Cross-domain (A→B)': {k: v for k, v in results.items()
                               if 'A→B' in k or 'DANN' in k or 'MMD' in k},
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (group_name, group_results) in zip(axes, groups.items(), strict=True):
        methods = list(group_results.keys())
        if not methods:
            continue

        x = np.arange(len(labels))
        width = 0.8 / max(len(methods), 1)

        for i, method in enumerate(methods):
            values = [group_results[method][m] for m in metrics]
            short_name = method.replace(' in-domain A', '').replace(' in-domain B', '')
            short_name = short_name.replace(' A→B', '').replace(' (baseline)', ' base')
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=short_name)

        ax.set_title(group_name, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Hodnota')
    plt.suptitle('Porovnanie domain adaptation metód', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graf uložený: {save_path}")
    plt.show()
