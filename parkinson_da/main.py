"""
Hlavný skript - semestrálny projekt, domain adaptation pre detekciu PD z reči.

Experimenty:
  1. In-domain (A, B, D) — baseline za každú doménu zvlášť
  2. Cross-domain bez DA (A→B, B→A) — horná hranica domain gapu
  3. Single-source DA (A→B): DANN, MMD, CORAL, CDAN, Contrastive
  4. Multi-source DA: (B+C)→D
  5. ROC krivky — všetky metódy pre A→B
  6. t-SNE — pred a po DANN (A→B)
  7. Bootstrap 95% CI — top metódy
  8. Záverečná porovnávacia tabuľka

Domény:
  A = Oxford (195 vzoriek, 31 pacientov, 12 akustických príznakov, 75% PD)
  B = Istanbul Recording 1 (80 vzoriek, 12+35 príznakov, 50% PD)
  C = Istanbul Recording 2 (80 vzoriek)
  D = Istanbul Recording 3 (80 vzoriek)
"""

import os
import time
from datetime import timedelta

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    BATCH_SIZE_DA,
    DATA_DIR,
    DEVICE,
    NUM_FEATURES,
    RANDOM_SEED,
)
from src.data import (
    create_cross_domain_loaders,
    create_loaders,
    create_multisource_loaders,
    download_datasets,
    load_domain,
    patient_wise_loaders,
)
from src.data.datasets import compute_class_weights
from src.evaluation import (
    bootstrap_ci,
    evaluate_model,
    evaluate_svm,
    extract_features,
    get_predictions,
    get_roc_data,
    print_comparison_table,
    print_metrics,
)
from src.models import (
    CDANModel,
    CNN1D,
    CORALModel,
    ContrastiveDAModel,
    DANNModel,
    MLP,
    MMDModel,
)
from src.training import (
    train_cdan,
    train_coral,
    train_contrastive,
    train_dann,
    train_mmd,
    train_model,
    train_multisource_dann,
)
from src.utils import (
    plot_comparison,
    plot_losses,
    plot_performance_drop,
    plot_roc_all_methods,
    plot_tsne,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helper: bežec jedného DA experimentu
# ---------------------------------------------------------------------------

def run_da_experiment(model, train_fn, src_loader, tgt_loader, tgt_test_loader,
                      model_type, name, results, losses_dict, train_kwargs=None):
    """Natrénuje DA model, evaluuje na cieľovej doméne."""
    losses = train_fn(model, src_loader, tgt_loader, **(train_kwargs or {}))
    losses_dict[name] = losses
    metrics = evaluate_model(model, tgt_test_loader, model_type=model_type)
    results[name] = metrics
    print_metrics(name, metrics)
    return model


def _src_class_weights(loader):
    """Vytéžované váhy tried zo zdrojového loaderu (kompenzuje 75% PD v Oxforde)."""
    ys = []
    for _, y in loader:
        ys.append(y.numpy())
    return compute_class_weights(np.concatenate(ys))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    print("=" * 70)
    print("SEMESTRÁLNY PROJEKT: Detekcia Parkinsonovej choroby z reči")
    print("Domain Adaptation — komplexné porovnanie techník")
    print("=" * 70)
    print(f"Zariadenie: {DEVICE}")
    set_seed(RANDOM_SEED)

    # ======================================================================
    # 1. PRÍPRAVA DÁT
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. PRÍPRAVA DÁT")
    print("=" * 70)

    required_files = [
        'oxford.csv', 'istanbul.csv',
        'istanbul_r1.csv', 'istanbul_r2.csv', 'istanbul_r3.csv',
    ]
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(DATA_DIR, f))]

    if missing:
        print(f"Chýbajúce súbory: {missing}")
        for fname in required_files:
            p = os.path.join(DATA_DIR, fname)
            if os.path.exists(p):
                os.remove(p)
        download_datasets()
    else:
        print("Všetky dátové súbory nájdené.")

    X_a, y_a, pids_a = load_domain('oxford')
    X_b, y_b, pids_b = load_domain('istanbul_r1')
    X_c, y_c, pids_c = load_domain('istanbul_r2')
    X_d, y_d, pids_d = load_domain('istanbul_r3')

    print(f"\nDoména A (Oxford):       {len(X_a):3d} vzoriek, "
          f"{len(set(pids_a)):2d} pacientov")
    print(f"Doména B (Istanbul R1):  {len(X_b):3d} vzoriek, "
          f"{len(set(pids_b)):2d} pacientov")
    print(f"Doména C (Istanbul R2):  {len(X_c):3d} vzoriek")
    print(f"Doména D (Istanbul R3):  {len(X_d):3d} vzoriek")

    # ======================================================================
    # 2. IN-DOMAIN EXPERIMENTY
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. IN-DOMAIN EXPERIMENTY")
    print("=" * 70)

    results = {}
    all_losses = {}

    # --- Doména A: patient-wise split (prevencia data leakage) ---
    print("\n--- Doména A (Oxford) — patient-wise split ---")
    tr_a, te_a, scaler_a = patient_wise_loaders(
        X_a, y_a, pids_a, test_size=0.25)

    print("\n[MLP] Doména A:")
    mlp_a = MLP(input_size=NUM_FEATURES)
    cw_a = _src_class_weights(tr_a)
    all_losses['MLP (A)'] = train_model(mlp_a, tr_a, class_weights=cw_a)
    results['MLP in-domain A'] = evaluate_model(mlp_a, te_a)
    print_metrics('MLP in-domain A', results['MLP in-domain A'])

    print("\n[CNN] Doména A:")
    cnn_a = CNN1D(input_size=NUM_FEATURES)
    all_losses['CNN (A)'] = train_model(cnn_a, tr_a, class_weights=cw_a)
    results['CNN in-domain A'] = evaluate_model(cnn_a, te_a)
    print_metrics('CNN in-domain A', results['CNN in-domain A'])

    print("\n[SVM] Doména A (patient-wise):")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx_a, test_idx_a = next(gss.split(X_a, y_a, groups=pids_a))
    svm_a = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED,
                    class_weight='balanced')),
    ])
    svm_a.fit(X_a[train_idx_a], y_a[train_idx_a])
    results['SVM in-domain A'] = evaluate_svm(svm_a, X_a[test_idx_a], y_a[test_idx_a])
    print_metrics('SVM in-domain A', results['SVM in-domain A'])

    # --- Doména B ---
    print("\n--- Doména B (Istanbul R1) ---")
    tr_b, te_b, scaler_b = create_loaders(
        X_b, y_b, test_size=0.2, batch_size=BATCH_SIZE)

    print("\n[MLP] Doména B:")
    mlp_b = MLP(input_size=NUM_FEATURES)
    all_losses['MLP (B)'] = train_model(mlp_b, tr_b)
    results['MLP in-domain B'] = evaluate_model(mlp_b, te_b)
    print_metrics('MLP in-domain B', results['MLP in-domain B'])

    print("\n[SVM] Doména B:")
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_b, y_b, test_size=0.2, random_state=RANDOM_SEED, stratify=y_b)
    svm_b = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED,
                    class_weight='balanced')),
    ])
    svm_b.fit(Xb_tr, yb_tr)
    results['SVM in-domain B'] = evaluate_svm(svm_b, Xb_te, yb_te)
    print_metrics('SVM in-domain B', results['SVM in-domain B'])

    # --- Doména D (cieľ pre multi-source) ---
    print("\n--- Doména D (Istanbul R3) ---")
    tr_d, te_d, _ = create_loaders(X_d, y_d, test_size=0.2, batch_size=BATCH_SIZE)
    print("\n[MLP] Doména D:")
    mlp_d = MLP(input_size=NUM_FEATURES)
    all_losses['MLP (D)'] = train_model(mlp_d, tr_d)
    results['MLP in-domain D'] = evaluate_model(mlp_d, te_d)
    print_metrics('MLP in-domain D', results['MLP in-domain D'])

    # ======================================================================
    # 3. CROSS-DOMAIN BASELINE
    # ======================================================================
    print("\n" + "=" * 70)
    print("3. CROSS-DOMAIN BASELINE (bez adaptácie)")
    print("=" * 70)

    # A→B
    print("\n--- A→B baseline ---")
    src_ab, tgt_ab_train, tgt_ab_test = create_cross_domain_loaders(
        X_a, y_a, X_b, y_b, batch_size=BATCH_SIZE_DA)

    print("\n[MLP] A→B baseline:")
    mlp_ab = MLP(input_size=NUM_FEATURES)
    cw_ab = _src_class_weights(src_ab)
    all_losses['MLP (A→B)'] = train_model(mlp_ab, src_ab, class_weights=cw_ab)
    results['MLP A→B (baseline)'] = evaluate_model(mlp_ab, tgt_ab_test)
    print_metrics('MLP A→B baseline', results['MLP A→B (baseline)'])

    print("\n[SVM] A→B baseline:")
    sc_src = StandardScaler()
    sc_tgt = StandardScaler()
    Xa_n = sc_src.fit_transform(X_a)
    Xb_n = sc_tgt.fit_transform(X_b)
    svm_ab = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED,
                 class_weight='balanced')
    svm_ab.fit(Xa_n, y_a)
    results['SVM A→B (baseline)'] = evaluate_svm(svm_ab, Xb_n, y_b)
    print_metrics('SVM A→B baseline', results['SVM A→B (baseline)'])

    # B→A (opačný smer)
    print("\n--- B→A baseline ---")
    src_ba, _, tgt_ba_test = create_cross_domain_loaders(
        X_b, y_b, X_a, y_a, batch_size=BATCH_SIZE_DA)
    mlp_ba = MLP(input_size=NUM_FEATURES)
    train_model(mlp_ba, src_ba)
    results['MLP B→A (baseline)'] = evaluate_model(mlp_ba, tgt_ba_test)
    print_metrics('MLP B→A baseline', results['MLP B→A (baseline)'])

    # ======================================================================
    # 4. DOMAIN ADAPTATION — A→B
    # ======================================================================
    print("\n" + "=" * 70)
    print("4. DOMAIN ADAPTATION — A→B (Oxford → Istanbul R1)")
    print("=" * 70)

    def fresh_ab():
        return create_cross_domain_loaders(
            X_a, y_a, X_b, y_b, batch_size=BATCH_SIZE_DA)

    print("\n--- DANN (Ganin & Lempitsky, 2015) ---")
    dann_ab = DANNModel(input_size=NUM_FEATURES)
    sl, tl, te = fresh_ab()
    dann_ab = run_da_experiment(dann_ab, train_dann, sl, tl, te,
                                'dann', 'DANN A→B', results, all_losses,
                                train_kwargs={'class_weights': cw_ab})

    print("\n--- MMD (Gretton et al., 2012) ---")
    mmd_ab = MMDModel(input_size=NUM_FEATURES)
    sl, tl, te = fresh_ab()
    mmd_ab = run_da_experiment(mmd_ab, train_mmd, sl, tl, te,
                               'mmd', 'MMD A→B', results, all_losses,
                               train_kwargs={'class_weights': cw_ab})

    print("\n--- CORAL (Sun & Saenko, 2016) ---")
    coral_ab = CORALModel(input_size=NUM_FEATURES)
    sl, tl, te = fresh_ab()
    coral_ab = run_da_experiment(coral_ab, train_coral, sl, tl, te,
                                 'coral', 'CORAL A→B', results, all_losses,
                                 train_kwargs={'class_weights': cw_ab})

    print("\n--- CDAN (Long et al., 2018) ---")
    cdan_ab = CDANModel(input_size=NUM_FEATURES)
    sl, tl, te = fresh_ab()
    cdan_ab = run_da_experiment(cdan_ab, train_cdan, sl, tl, te,
                                'cdan', 'CDAN A→B', results, all_losses,
                                train_kwargs={'class_weights': cw_ab})

    print("\n--- Contrastive DA (Yang et al., 2021) ---")
    cont_ab = ContrastiveDAModel(input_size=NUM_FEATURES)
    sl, tl, te = fresh_ab()
    cont_ab = run_da_experiment(cont_ab, train_contrastive, sl, tl, te,
                                'contrastive', 'Contrastive A→B', results, all_losses,
                                train_kwargs={'class_weights': cw_ab})

    # ======================================================================
    # 5. MULTI-SOURCE DA — (B+C)→D
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. MULTI-SOURCE DA — (B+C)→D")
    print("=" * 70)

    source_list = [(X_b, y_b, 0), (X_c, y_c, 1)]

    print("\n--- Multi-source DANN (B+C)→D ---")
    dann_ms = DANNModel(input_size=NUM_FEATURES, num_domains=3)
    sl_ms, tl_ms, te_ms = create_multisource_loaders(source_list, X_d, y_d)
    losses_ms = train_multisource_dann(dann_ms, sl_ms, tl_ms)
    all_losses['Multi-DANN (B+C→D)'] = losses_ms
    results['Multi-DANN (B+C)→D'] = evaluate_model(dann_ms, te_ms, model_type='dann')
    print_metrics('Multi-DANN (B+C)→D', results['Multi-DANN (B+C)→D'])

    print("\n--- Multi-source CORAL (B+C)→D ---")
    coral_ms = CORALModel(input_size=NUM_FEATURES)
    sl_ms2, tl_ms2, te_ms2 = create_multisource_loaders(source_list, X_d, y_d)
    losses_ms_coral = train_coral(coral_ms, sl_ms2, tl_ms2)
    all_losses['Multi-CORAL (B+C→D)'] = losses_ms_coral
    results['Multi-CORAL (B+C)→D'] = evaluate_model(coral_ms, te_ms2, model_type='coral')
    print_metrics('Multi-CORAL (B+C)→D', results['Multi-CORAL (B+C)→D'])

    # ======================================================================
    # 6. ROC KRIVKY — A→B
    # ======================================================================
    print("\n" + "=" * 70)
    print("6. ROC KRIVKY — A→B")
    print("=" * 70)

    # Celá Istanbul R1 normalizovaná pre konzistentné ROC porovnanie
    sc_roc = StandardScaler().fit(X_a)   # fitujeme na zdrojovej doméne
    Xb_roc_arr = StandardScaler().fit_transform(X_b)
    tgt_roc_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xb_roc_arr), torch.LongTensor(y_b)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    roc_models = {
        'MLP (baseline)': (mlp_ab,  'standard'),
        'DANN':           (dann_ab,  'dann'),
        'MMD':            (mmd_ab,   'mmd'),
        'CORAL':          (coral_ab, 'coral'),
        'CDAN':           (cdan_ab,  'cdan'),
        'Contrastive':    (cont_ab,  'contrastive'),
    }

    roc_data = {}
    for name, (model, mtype) in roc_models.items():
        fpr, tpr, auc_score = get_roc_data(model, tgt_roc_loader, model_type=mtype)
        roc_data[name] = (fpr, tpr, auc_score)
        print(f"  {name:<20s}: AUC = {auc_score:.3f}")

    plot_roc_all_methods(
        roc_data,
        title="ROC krivky — A→B (Oxford → Istanbul R1)",
        save_path=os.path.join(DATA_DIR, 'roc_curves_AtoB.png')
    )

    # ======================================================================
    # 7. t-SNE VIZUALIZÁCIA — pred a po DANN
    # ======================================================================
    print("\n" + "=" * 70)
    print("7. t-SNE — pred a po DANN (A→B)")
    print("=" * 70)

    Xa_norm_arr = StandardScaler().fit_transform(X_a)
    src_tsne_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xa_norm_arr), torch.LongTensor(y_a)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    dann_before = DANNModel(input_size=NUM_FEATURES)  # náhodná inicializácia
    feats_sb, labs_sb = extract_features(dann_before, src_tsne_loader, 'dann')
    feats_tb, labs_tb = extract_features(dann_before, tgt_roc_loader, 'dann')
    feats_sa, labs_sa = extract_features(dann_ab, src_tsne_loader, 'dann')
    feats_ta, labs_ta = extract_features(dann_ab, tgt_roc_loader, 'dann')

    if all(f is not None for f in [feats_sb, feats_tb, feats_sa, feats_ta]):
        plot_tsne(
            features_dict={
                'Pred DANN': (feats_sb, feats_tb),
                'Po DANN':   (feats_sa, feats_ta),
            },
            labels_dict={
                'Pred DANN': (labs_sb, labs_tb),
                'Po DANN':   (labs_sa, labs_ta),
            },
            title="t-SNE: Feature Space A→B pred a po DANN",
            save_path=os.path.join(DATA_DIR, 'tsne_dann.png')
        )

    # ======================================================================
    # 8. BOOTSTRAP 95% CI
    # ======================================================================
    print("\n" + "=" * 70)
    print("8. BOOTSTRAP 95% CI — AUC pre A→B")
    print("=" * 70)
    print(f"\n  {'Metóda':<22} {'AUC':>6}  {'95% CI':>15}")
    print("  " + "-" * 46)
    for name, (model, mtype) in roc_models.items():
        y_true_ci, y_score_ci = get_predictions(model, tgt_roc_loader, mtype)
        ci_lo, ci_hi = bootstrap_ci(y_true_ci, y_score_ci)
        auc_val = roc_data[name][2]
        print(f"  {name:<22} {auc_val:>6.3f}  [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ======================================================================
    # 9. PERFORMANCE DROP
    # ======================================================================
    print("\n" + "=" * 70)
    print("9. PERFORMANCE DROP — domain gap")
    print("=" * 70)

    plot_performance_drop(
        {'MLP': results.get('MLP in-domain A', {}),
         'SVM': results.get('SVM in-domain A', {})},
        {'MLP': results.get('MLP A→B (baseline)', {}),
         'SVM': results.get('SVM A→B (baseline)', {})},
        metric='auc',
        title="Domain Gap: In-domain A vs Cross-domain A→B",
        save_path=os.path.join(DATA_DIR, 'performance_drop.png')
    )

    # ======================================================================
    # 10. ZÁVEREČNÁ TABUĽKA
    # ======================================================================
    print("\n" + "=" * 70)
    print("10. ZÁVEREČNÁ POROVNÁVACIA TABUĽKA")
    print("=" * 70)
    print_comparison_table(results)

    da_losses = {k: v for k, v in all_losses.items()
                 if any(x in k for x in ('DANN', 'MMD', 'CORAL', 'CDAN', 'Contrastive'))}
    if da_losses:
        plot_losses(da_losses, "Trénovacie straty DA metód",
                    save_path=os.path.join(DATA_DIR, 'losses_da.png'))

    print(f"\nCelkový čas: {timedelta(seconds=int(time.time() - start_time))}")
    print("Grafy uložené v:", DATA_DIR)


if __name__ == '__main__':
    main()


