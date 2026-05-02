"""
Robustný multi-seed experiment pre štatisticky obhájiteľné výsledky.

Motivácia:
  Pôvodný main.py beží s jediným seedom (42) na jedinom train/test splitte.
  Pri test sete s 80 vzorkami a Δ AUC ~0.05 je 95% CI [0.45, 0.74] tak široký,
  že nedokážeme rozhodnúť, či je nejaká DA metóda lepšia ako baseline.

  Riešenie: každú metódu spustíme s N=10 rôznymi seedmi (rôzna PyTorch
  inicializácia + rôzne mini-batch poradia), ALE EVALUUJEME NA ROVNAKOM
  test sete cieľovej domény. Toto umožňuje:
   1. mean ± std AUC zo 10 nezávislých tréningov  (variance odhadu modelu)
   2. paired bootstrap CI na Δ AUC = method - baseline na rovnakom teste
      → eliminuje zdieľanú variance test sample-u  → tesnejší interval
   3. p-value pre H0: Δ ≤ 0 cez kombinované bootstrap iterácie zo všetkých seedov.

Experimenty:
  E1. A → Istanbul-pooled (B+C+D, 240 vz., 80 jedinečných pacientov)
      patient-wise rozdelené na 60% train target (unsup) / 40% test
  E2. Istanbul-pooled-train → A
      Istanbul ako zdroj, Oxford ako cieľ (opačný smer)

Metódy: Source-only (baseline) vs DANN, MMD, CORAL, CDAN, Contrastive,
        Subspace Alignment (closed-form klasický baseline).

Tabuľky a CSV sa ukladajú do data/robust_results/.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE_DA,
    DATA_DIR,
    DEVICE,
    NUM_FEATURES,
    SEEDS,
)
from src.data import download_datasets, load_domain
from src.data.datasets import ParkinsonDataset, compute_class_weights
from src.evaluation import (
    paired_bootstrap_diff,
)
from src.models import (
    CDANModel,
    CORALModel,
    ContrastiveDAModel,
    DANNModel,
    MLP,
    MMDModel,
    SubspaceAlignmentDA,
)
from src.training import (
    train_cdan,
    train_contrastive,
    train_coral,
    train_dann,
    train_mmd,
    train_model,
)
from src.utils import set_seed
from sklearn.metrics import roc_auc_score


# =============================================================================
# Pomocné: pripravenie cross-domain loaderov pre konkrétny seed a split
# =============================================================================

def _make_loaders(X_src, y_src, X_tgt_train, y_tgt_train, X_tgt_test, y_tgt_test,
                  seed, batch_size=BATCH_SIZE_DA):
    """Source = celý zdroj. Target_train = unlab. časť cieľa pre DA. Test = nedotknutý cieľ."""
    sc_src = StandardScaler().fit(X_src)
    sc_tgt = StandardScaler().fit(X_tgt_train)  # škálovanie z dostupných target dát
    Xs = sc_src.transform(X_src).astype(np.float32)
    Xt_tr = sc_tgt.transform(X_tgt_train).astype(np.float32)
    Xt_te = sc_tgt.transform(X_tgt_test).astype(np.float32)

    g = torch.Generator().manual_seed(seed)
    src_loader = DataLoader(ParkinsonDataset(Xs, y_src),
                            batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=g)
    tgt_loader = DataLoader(ParkinsonDataset(Xt_tr, y_tgt_train),
                            batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=g)
    test_loader = DataLoader(ParkinsonDataset(Xt_te, y_tgt_test),
                             batch_size=batch_size * 2, shuffle=False)
    return src_loader, tgt_loader, test_loader, Xt_te


# =============================================================================
# Skórovanie modelu na test loaderi → vráti y_score (PD prob)
# =============================================================================

def _scores(model, test_loader, model_type='standard'):
    model = model.to(DEVICE).eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            if model_type in ('dann', 'cdan'):
                logits, _, _ = model(x)
            elif model_type in ('mmd', 'coral', 'contrastive'):
                logits, _ = model(x)
            else:
                logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ps.append(p)
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# =============================================================================
# Trénery — wrapper, ktorý pre danú konfiguráciu vráti y_score
# =============================================================================

def _run_method(method, X_src, y_src, X_tgt_train, y_tgt_train,
                X_tgt_test, y_tgt_test, seed, class_weights):
    set_seed(seed)
    sl, tl, te, Xt_te = _make_loaders(
        X_src, y_src, X_tgt_train, y_tgt_train, X_tgt_test, y_tgt_test, seed)

    if method == 'baseline':
        m = MLP(input_size=NUM_FEATURES)
        train_model(m, sl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'standard')
    elif method == 'dann':
        m = DANNModel(input_size=NUM_FEATURES)
        train_dann(m, sl, tl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'dann')
    elif method == 'mmd':
        m = MMDModel(input_size=NUM_FEATURES)
        train_mmd(m, sl, tl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'mmd')
    elif method == 'coral':
        m = CORALModel(input_size=NUM_FEATURES)
        train_coral(m, sl, tl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'coral')
    elif method == 'cdan':
        m = CDANModel(input_size=NUM_FEATURES)
        train_cdan(m, sl, tl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'cdan')
    elif method == 'contrastive':
        m = ContrastiveDAModel(input_size=NUM_FEATURES)
        train_contrastive(m, sl, tl, class_weights=class_weights)
        y_true, y_score = _scores(m, te, 'contrastive')
    elif method == 'subspace':
        # Closed-form, ale rebalancujeme zdroj cez SVC class_weight='balanced'
        sa = SubspaceAlignmentDA(n_components=8, random_state=seed)
        sa.fit(X_src, y_src, X_tgt_train)
        y_true = y_tgt_test
        y_score = sa.predict_proba(X_tgt_test)[:, 1]
    elif method == 'svm_source':
        sc = StandardScaler().fit(X_src)
        Xs = sc.transform(X_src)
        Xt = StandardScaler().fit(X_tgt_train).transform(X_tgt_test)
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced',
                  random_state=seed)
        clf.fit(Xs, y_src)
        y_true = y_tgt_test
        y_score = clf.predict_proba(Xt)[:, 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    return y_true, y_score


# =============================================================================
# Patient-wise pooled split pre Istanbul (B+C+D)
# =============================================================================

def _pooled_istanbul():
    """Vráti X, y, patient_ids spojené z B+C+D (240 záznamov, 80 jedinečných pac.)."""
    Xs, ys, pids = [], [], []
    for sess in ('istanbul_r1', 'istanbul_r2', 'istanbul_r3'):
        X, y, p = load_domain(sess)
        Xs.append(X)
        ys.append(y)
        # Patient IDs sú už zdieľané (rovnakí pacienti naprieč session-mi),
        # ale aby sme neriskovali kolíziu indexov, zachováme ich tak ako sú.
        pids.append(p)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(pids)


# =============================================================================
# Hlavný experiment
# =============================================================================

def run_experiment(name, X_src, y_src, X_tgt_full, y_tgt_full, pids_tgt,
                   methods, n_seeds=10, target_test_size=0.4,
                   patient_wise_target=True):
    """Pre každú metódu N seedov → uloží y_score, vypočíta paired bootstrap Δ vs baseline."""
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"  Source: {len(X_src)} vz., Target: {len(X_tgt_full)} vz., "
          f"{len(np.unique(pids_tgt))} unikátnych pacientov")

    # Patient-wise rozdelenie cieľa: časť pre unsup. DA train, časť pre test
    if patient_wise_target and pids_tgt is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=target_test_size,
                                random_state=12345)
        tr_idx, te_idx = next(gss.split(X_tgt_full, y_tgt_full, groups=pids_tgt))
    else:
        from sklearn.model_selection import train_test_split
        tr_idx, te_idx = train_test_split(
            np.arange(len(X_tgt_full)),
            test_size=target_test_size,
            random_state=12345,
            stratify=y_tgt_full,
        )
    X_tgt_tr = X_tgt_full[tr_idx]
    y_tgt_tr = y_tgt_full[tr_idx]
    X_tgt_te = X_tgt_full[te_idx]
    y_tgt_te = y_tgt_full[te_idx]

    # source class weights raz
    cw = compute_class_weights(y_src)
    print(f"  Source class weights: {cw.numpy().round(3).tolist()}")
    print(f"  Target test: {len(y_tgt_te)} vz., PD={int(y_tgt_te.sum())}, "
          f"H={int((1-y_tgt_te).sum())}")

    # uložené predikcie: {method: list of (y_true, y_score) per seed}
    preds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in methods}

    for seed in SEEDS[:n_seeds]:
        for method in methods:
            t0 = time.time()
            y_true, y_score = _run_method(
                method, X_src, y_src, X_tgt_tr, y_tgt_tr,
                X_tgt_te, y_tgt_te, seed, cw if method not in ('subspace', 'svm_source') else None
            )
            preds[method].append((y_true, y_score))
            auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float('nan')
            print(f"  [seed={seed:>4}] {method:<14s} AUC={auc:.3f}  ({time.time()-t0:.1f}s)")

    # ==== Agregácia: mean ± std a paired bootstrap Δ vs baseline ====
    rows = []
    baseline_preds = preds['baseline']
    for method in methods:
        aucs = []
        for y, s in preds[method]:
            if len(np.unique(y)) > 1:
                aucs.append(roc_auc_score(y, s))
        aucs = np.asarray(aucs)
        row = {
            'method': method,
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            'auc_min': float(np.min(aucs)),
            'auc_max': float(np.max(aucs)),
            'n_seeds': len(aucs),
        }
        if method != 'baseline':
            # Spojený paired bootstrap: pre každý seed paired Δ a CI
            # → kombinujeme cez agregáciu rozdielov všetkých seedov
            #   pomocou paired-bootstrap pre KAŽDÝ seed-pair, potom mean Δ a
            #   kombinovaný CI cez konkatenáciu bootstrap delta arrays.
            all_diffs = []
            point_diffs = []
            for (y_b, s_b), (y_m, s_m) in zip(baseline_preds, preds[method]):
                # Predpokladáme, že y_b == y_m (rovnaký test set, rovnaký seed)
                d_pt, _, _, _ = paired_bootstrap_diff(
                    y_b, s_b, s_m, n_bootstrap=200, random_state=42
                )
                # Pre kombinované CI ručne získame raw bootstrap pole
                rng = np.random.RandomState(42)
                n = len(y_b)
                seed_diffs = []
                for _ in range(200):
                    idx = rng.randint(0, n, n)
                    if len(np.unique(y_b[idx])) < 2:
                        continue
                    seed_diffs.append(
                        roc_auc_score(y_m[idx], s_m[idx]) -
                        roc_auc_score(y_b[idx], s_b[idx])
                    )
                all_diffs.extend(seed_diffs)
                point_diffs.append(d_pt)

            all_diffs_arr = np.asarray(all_diffs)
            mean_delta = float(np.mean(point_diffs))
            ci_lo = float(np.quantile(all_diffs_arr, 0.025))
            ci_hi = float(np.quantile(all_diffs_arr, 0.975))
            # one-sided p-value: H0: Δ ≤ 0
            p_one = float(np.mean(all_diffs_arr <= 0))
            row.update({
                'delta_vs_baseline': mean_delta,
                'delta_ci_low': ci_lo,
                'delta_ci_high': ci_hi,
                'p_value_one_sided': p_one,
                'significant_005': bool(p_one < 0.05 and ci_lo > 0),
            })
        else:
            row.update({
                'delta_vs_baseline': 0.0,
                'delta_ci_low': 0.0,
                'delta_ci_high': 0.0,
                'p_value_one_sided': float('nan'),
                'significant_005': False,
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n  Súhrn metód (mean ± std AUC, paired bootstrap Δ vs baseline):")
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    pd.set_option('display.width', 140)
    print(df.to_string(index=False))
    return df, preds


def main():
    t0 = time.time()
    print(f"Zariadenie: {DEVICE}")
    print(f"SEEDS: {SEEDS}")

    # ==== Načítanie dát ====
    required_files = ['oxford.csv', 'istanbul_r1.csv', 'istanbul_r2.csv', 'istanbul_r3.csv']
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"Chýbajúce: {missing} → sťahujem...")
        download_datasets()

    X_a, y_a, pids_a = load_domain('oxford')
    X_pool, y_pool, pids_pool = _pooled_istanbul()
    print(f"\nA (Oxford):           {len(X_a)} vz., "
          f"PD={int(y_a.sum())}, H={int((1-y_a).sum())}, "
          f"{len(np.unique(pids_a))} pacientov")
    print(f"Istanbul-pooled (BCD): {len(X_pool)} vz., "
          f"PD={int(y_pool.sum())}, H={int((1-y_pool).sum())}, "
          f"{len(np.unique(pids_pool))} pacientov")

    methods = ['baseline', 'svm_source', 'subspace',
               'dann', 'mmd', 'coral', 'cdan', 'contrastive']
    n_seeds = 10

    out_dir = os.path.join(DATA_DIR, 'robust_results')
    os.makedirs(out_dir, exist_ok=True)

    # ==== E1: A → Istanbul-pooled ====
    df_e1, preds_e1 = run_experiment(
        'E1: A (Oxford) → Istanbul-pooled (B+C+D)',
        X_a, y_a, X_pool, y_pool, pids_pool,
        methods=methods, n_seeds=n_seeds,
        target_test_size=0.4, patient_wise_target=True,
    )
    df_e1.to_csv(os.path.join(out_dir, 'E1_A_to_IstanbulPooled.csv'), index=False)

    # ==== E2: Istanbul-pooled → A ====
    df_e2, preds_e2 = run_experiment(
        'E2: Istanbul-pooled (B+C+D) → A (Oxford)',
        X_pool, y_pool, X_a, y_a, pids_a,
        methods=methods, n_seeds=n_seeds,
        target_test_size=0.4, patient_wise_target=True,
    )
    df_e2.to_csv(os.path.join(out_dir, 'E2_IstanbulPooled_to_A.csv'), index=False)

    # ==== Spojený výpis ====
    print("\n" + "=" * 72)
    print("FINÁLNE VÝSLEDKY — multi-seed (N=10) s paired bootstrap CI")
    print("=" * 72)
    for name, df in [('E1: A → Istanbul-pooled', df_e1),
                     ('E2: Istanbul-pooled → A', df_e2)]:
        print(f"\n{name}")
        print("-" * 72)
        cols = ['method', 'auc_mean', 'auc_std', 'delta_vs_baseline',
                'delta_ci_low', 'delta_ci_high', 'p_value_one_sided',
                'significant_005']
        print(df[cols].to_string(index=False))

    print(f"\nCSV výstupy uložené v: {out_dir}")
    print(f"Celkový čas: {timedelta(seconds=int(time.time() - t0))}")


if __name__ == '__main__':
    main()
