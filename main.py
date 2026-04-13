"""
Hlavný skript - spustenie celého experimentu.

Použitie:
    python main.py

Vzor štruktúry: Cvičenie 4-5 (kompletný workflow)
  imports → device → dáta → model → trénovanie → testovanie

Tento skript realizuje celý experiment:
  Týždeň 1-2: Stiahnutie a príprava dát
  Týždeň 3-5: Trénovanie klasifikátorov (MLP, CNN, SVM)
  Týždeň 6:   In-domain evaluácia
  Týždeň 7-8: Domain adaptácia (DANN, MMD) + cross-domain evaluácia
"""

import os
import time
from datetime import timedelta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from config import DEVICE, DATA_DIR, RANDOM_SEED
from download_data import download_datasets
from datasets import load_domain, create_loaders, create_cross_domain_loaders
from models import MLP, CNN1D
from domain_adaptation import DANNModel, MMDModel
from train import train_model, train_dann, train_mmd
from evaluate import (evaluate_model, evaluate_svm,
                      print_metrics, print_comparison_table)
from utils import set_seed, plot_losses, plot_comparison


def main():
    start_time = time.time()

    print("=" * 60)
    print("SEMESTRÁLNY PROJEKT: Detekcia Parkinsonovej choroby z reči")
    print("Domain Adaptation - porovnanie techník")
    print("=" * 60)
    print(f"Zariadenie: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")
    set_seed(RANDOM_SEED)

    # ==================================================================
    # TÝŽDEŇ 1-2: Príprava dát
    # Vzor: Cvičenie 1-2 (tensory, práca s dátami)
    # ==================================================================
    print("\n" + "=" * 60)
    print("TÝŽDEŇ 1-2: Príprava dát")
    print("=" * 60)

    # Skontrolujeme, či už máme dáta stiahnuté
    oxford_path = os.path.join(DATA_DIR, 'oxford.csv')
    istanbul_path = os.path.join(DATA_DIR, 'istanbul.csv')

    if not (os.path.exists(oxford_path) and os.path.exists(istanbul_path)):
        print("Dáta nenájdené, sťahujem...")
        download_datasets()
    else:
        print("Dáta už existujú, preskakujem sťahovanie.")

    # Načítanie dát
    X_oxford, y_oxford = load_domain('oxford')
    X_istanbul, y_istanbul = load_domain('istanbul')
    print(f"\nDomain A (Oxford):   {len(X_oxford)} vzoriek")
    print(f"Domain B (Istanbul): {len(X_istanbul)} vzoriek")

    # ==================================================================
    # TÝŽDEŇ 3-5: Trénovanie klasifikátorov (in-domain)
    # Vzor: Cvičenie 4-5 (MLP na MNIST, CNN na CIFAR10)
    # ==================================================================
    print("\n" + "=" * 60)
    print("TÝŽDEŇ 3-6: In-domain trénovanie a evaluácia")
    print("=" * 60)

    results = {}
    all_losses = {}

    # ---- In-domain: Domain A (Oxford) ----
    print("\n--- Domain A (Oxford) - In-domain ---")
    train_loader_a, test_loader_a, scaler_a = create_loaders(X_oxford, y_oxford)

    # MLP na Domain A
    print("\n[MLP] Trénovanie na Domain A:")
    mlp_a = MLP()
    losses_mlp_a = train_model(mlp_a, train_loader_a)
    metrics = evaluate_model(mlp_a, test_loader_a)
    results['MLP in-domain A'] = metrics
    all_losses['MLP (Domain A)'] = losses_mlp_a
    print_metrics('MLP in-domain A', metrics)

    # CNN na Domain A
    print("\n[CNN] Trénovanie na Domain A:")
    cnn_a = CNN1D()
    losses_cnn_a = train_model(cnn_a, train_loader_a)
    metrics = evaluate_model(cnn_a, test_loader_a)
    results['CNN in-domain A'] = metrics
    all_losses['CNN (Domain A)'] = losses_cnn_a
    print_metrics('CNN in-domain A', metrics)

    # SVM na Domain A
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    print("\n[SVM] Trénovanie na Domain A:")
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        X_oxford, y_oxford, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y_oxford
    )
    svm_a = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    svm_a.fit(Xa_train, ya_train)
    metrics = evaluate_svm(svm_a, Xa_test, ya_test)
    results['SVM in-domain A'] = metrics
    print_metrics('SVM in-domain A', metrics)

    # ---- In-domain: Domain B (Istanbul) ----
    print("\n--- Domain B (Istanbul) - In-domain ---")
    train_loader_b, test_loader_b, scaler_b = create_loaders(
        X_istanbul, y_istanbul
    )

    # MLP na Domain B
    print("\n[MLP] Trénovanie na Domain B:")
    mlp_b = MLP()
    losses_mlp_b = train_model(mlp_b, train_loader_b)
    metrics = evaluate_model(mlp_b, test_loader_b)
    results['MLP in-domain B'] = metrics
    all_losses['MLP (Domain B)'] = losses_mlp_b
    print_metrics('MLP in-domain B', metrics)

    # CNN na Domain B
    print("\n[CNN] Trénovanie na Domain B:")
    cnn_b = CNN1D()
    losses_cnn_b = train_model(cnn_b, train_loader_b)
    metrics = evaluate_model(cnn_b, test_loader_b)
    results['CNN in-domain B'] = metrics
    all_losses['CNN (Domain B)'] = losses_cnn_b
    print_metrics('CNN in-domain B', metrics)

    # SVM na Domain B
    print("\n[SVM] Trénovanie na Domain B:")
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_istanbul, y_istanbul, test_size=0.2,
        random_state=RANDOM_SEED, stratify=y_istanbul
    )
    svm_b = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    svm_b.fit(Xb_train, yb_train)
    metrics = evaluate_svm(svm_b, Xb_test, yb_test)
    results['SVM in-domain B'] = metrics
    print_metrics('SVM in-domain B', metrics)

    # ==================================================================
    # TÝŽDEŇ 7-8: Cross-domain evaluácia a Domain Adaptácia
    # ==================================================================
    print("\n" + "=" * 60)
    print("TÝŽDEŇ 7-8: Cross-domain evaluácia + Domain Adaptation")
    print("=" * 60)

    # ---- Cross-domain BEZ adaptácie (baseline) ----
    # Trénovanie na Domain A, testovanie na Domain B
    print("\n--- Cross-domain: Train A → Test B (bez adaptácie) ---")

    # Vytvoríme zdieľané loadery pre cross-domain experimenty
    source_loader, target_loader, _ = create_cross_domain_loaders(
        X_oxford, y_oxford, X_istanbul, y_istanbul
    )

    print("\n[MLP] Baseline (bez adaptácie):")
    mlp_cross = MLP()
    train_model(mlp_cross, source_loader)
    metrics = evaluate_model(mlp_cross, target_loader)
    results['MLP A→B (baseline)'] = metrics
    print_metrics('MLP A→B baseline', metrics)

    # CNN cross-domain
    print("\n[CNN] Baseline (bez adaptácie):")
    cnn_cross = CNN1D()
    train_model(cnn_cross, source_loader)
    metrics = evaluate_model(cnn_cross, target_loader)
    results['CNN A→B (baseline)'] = metrics
    print_metrics('CNN A→B baseline', metrics)

    # SVM cross-domain
    print("\n[SVM] Baseline (bez adaptácie):")
    svm_cross = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    svm_cross.fit(X_oxford, y_oxford)
    metrics = evaluate_svm(svm_cross, X_istanbul, y_istanbul)
    results['SVM A→B (baseline)'] = metrics
    print_metrics('SVM A→B baseline', metrics)

    # ---- DANN ----
    # Ref: Ganin et al. (2015), https://arxiv.org/abs/1409.7495
    print("\n--- DANN: Domain-Adversarial Neural Network ---")
    print("  Ref: Ganin & Lempitsky (2015)")

    dann_model = DANNModel()
    source_loader_dann, target_loader_dann, _ = create_cross_domain_loaders(
        X_oxford, y_oxford, X_istanbul, y_istanbul
    )
    losses_dann = train_dann(dann_model, source_loader_dann, target_loader_dann)
    all_losses['DANN'] = losses_dann
    metrics = evaluate_model(dann_model, target_loader_dann, model_type='dann')
    results['DANN A→B'] = metrics
    print_metrics('DANN A→B', metrics)

    # ---- MMD ----
    # Ref: Gretton et al. (2012)
    print("\n--- MMD: Maximum Mean Discrepancy ---")
    print("  Ref: Gretton et al. (2012)")

    mmd_model = MMDModel()
    source_loader_mmd, target_loader_mmd, _ = create_cross_domain_loaders(
        X_oxford, y_oxford, X_istanbul, y_istanbul
    )
    losses_mmd = train_mmd(mmd_model, source_loader_mmd, target_loader_mmd)
    all_losses['MMD'] = losses_mmd
    metrics = evaluate_model(mmd_model, target_loader_mmd, model_type='mmd')
    results['MMD A→B'] = metrics
    print_metrics('MMD A→B', metrics)

    # ==================================================================
    # Výsledky
    # ==================================================================
    print_comparison_table(results)

    # Vizualizácia
    print("\nVykresľujem grafy...")
    plot_losses(all_losses, "Trénovacie straty",
                save_path=os.path.join(DATA_DIR, 'losses.png'))
    plot_comparison(results,
                    save_path=os.path.join(DATA_DIR, 'comparison.png'))

    elapsed = timedelta(seconds=time.time() - start_time)
    print(f"\nCelkový čas: {elapsed}")

    # ==================================================================
    # TODO: Týždeň 9-13 (budúce práce)
    # ==================================================================
    print("\n" + "=" * 60)
    print("TODO: Plán ďalších týždňov")
    print("=" * 60)
    print("""
  Týždeň 9  (13-19 Apr): CORAL implementácia
    - Deep CORAL (Sun & Saenko, 2016)
    - Zarovnanie kovariančných matíc medzi doménami

  Týždeň 10 (20-26 Apr): Multi-source domain adaptation
    - Trénovanie na oboch doménach súčasne
    - Váženie zdrojových domén

  Týždeň 11 (27 Apr - 3 May): Kompletná evaluácia
    - Oba smery: A→B aj B→A
    - Štatistická významnosť (bootstrap)
    - Analýza doménového posunu (t-SNE vizualizácia)

  Týždeň 12 (4-10 May): Vizualizácia a analýza
    - Porovnávacie tabuľky pre všetky kombinácie
    - ROC krivky
    - Analýza chýb (false positives/negatives)

  Týždeň 13 (11-17 May): Záverečná správa a prezentácia
    - Zhodnotenie najlepšej kombinácie klasifikátor + DA
    - Odporúčania pre klinické nasadenie
    - Príprava prezentácie
    """)


if __name__ == '__main__':
    main()
