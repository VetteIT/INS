# PROJECT CONTEXT — Expert Analysis File
## Parkinson's Disease Detection via Domain Adaptation

> **Purpose of this file:** Provide complete, expert-level context about this project
> for an AI assistant continuing work on it. This file contains no prescriptive
> directions — only facts, observations, and structured context from which an
> independent analysis can be formed.

---

## 1. Assignment Requirements (Original Task)

The project must fulfill the following academic requirements:

### 1.1 Classifier types (must implement ≥ 1)
- CNN model over spectrograms
- Traditional model over extracted acoustic features (MFCC, jitter, shimmer) + SVM/MLP
- Foundation model for speech (wav2vec2, HuBERT, Whisper over spectrogram)

### 1.2 Datasets as domains (from the suggested list)
- **PDITA** — suggested in assignment
- **Neurovoz** — suggested in assignment
- **PC-GITA** — suggested in assignment
*(Note: the current implementation uses Oxford UCI and Istanbul UCI instead — see section 3)*

### 1.3 Domain Adaptation methods (must compare multiple)
- No adaptation (baseline)
- DANN — Domain-Adversarial Neural Network
- Multi-source domain adaptation
- Contrastive domain alignment
- Moment matching (MMD)

### 1.4 Evaluation setting
- Train on one or more source domains, test on an **unseen target domain without annotations**
- Compare classification metrics: AUC, F1-score, sensitivity/specificity
- Analyze performance drop between in-domain and out-of-domain

### 1.5 Final goal
Determine which combination of classifier + DA technique provides the best
generalization for **real clinical deployment**.

---

## 2. Current Implementation Status

### 2.1 What IS implemented
| Requirement | Status | Notes |
|---|---|---|
| MLP classifier | ✅ | 12→128→64→2, dropout=0.3 |
| CNN1D classifier | ✅ | 2× Conv1d + MaxPool, over 12 acoustic features |
| SVM classifier | ✅ | RBF kernel, class_weight='balanced', sklearn Pipeline |
| DANN | ✅ | GRL with sigmoid λ schedule, inverse LR |
| MMD | ✅ | Multi-kernel RBF, median heuristic bandwidth |
| CORAL | ✅ | Frobenius norm of covariance difference |
| CDAN | ✅ | Class-conditional (feature ⊗ softmax) domain classifier |
| Contrastive DA | ✅ | NT-Xent / prototype-based with confidence threshold 0.75 |
| Multi-source DA | ✅ | (B+C)→D using DANN and CORAL |
| Patient-wise split | ✅ | GroupShuffleSplit prevents data leakage |
| ROC curves | ✅ | All A→B methods on one plot |
| t-SNE visualization | ✅ | Before/after DANN A→B |
| Bootstrap 95% CI | ✅ | N=1000 iterations on AUC |
| Performance drop chart | ✅ | In-domain vs cross-domain bar chart |

### 2.2 What is NOT implemented vs. assignment
| Missing | Notes                                                                                                                                                   |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| CNN over **spectrograms** | Current CNN1D works over scalar acoustic features, NOT raw audio/spectrograms                                                                           |
| Foundation model (wav2vec2 / HuBERT / Whisper) | Not implemented — would require raw audio files                                                                                                         |
| **PDITA / Neurovoz / PC-GITA** datasets | Assignment specifically names these, but WE CAN USE ANY OF POSSIBLE DATASETS; current implementation uses Oxford UCI #174 and Istanbul UCI #489 instead |
| Cross-domain test without ANY target labels | Currently target test set labels ARE used for evaluation (correct), but domain alignment is unsupervised — this is correct UDA setting                  |

---

## 3. Datasets Used — Detailed Characterization

### 3.1 Domain A — Oxford Parkinson's Dataset
- **Source:** UCI ML Repository #174, Little et al. (2008)
- **URL:** https://archive.ics.uci.edu/dataset/174/parkinsons
- **Size:** 195 samples, 32 subjects (S01–S50, ~6 recordings each)
- **Class balance:** 147 PD (75.4%) vs 48 Healthy (24.6%) — **severely imbalanced**
- **Recording protocol:** Sustained phonation of vowel /a/, single microphone,
  MDVP software (Kay Pentax)
- **Features (12):** Jitter (Rel, Abs, RAP, PPQ), Shimmer (loc, dB, APQ3, APQ5),
  HNR, RPDE, DFA, PPE
- **Patient split used:** GroupShuffleSplit by subject ID (S01–S50),
  25% held out → ~8 subjects test (~48 samples)
- **Known issues:**
  - Extreme class imbalance (3:1 PD:Healthy)
  - Small absolute number of healthy subjects (only ~8)
  - Single recording protocol, single microphone, single lab
  - HNR is MDVP broadband HNR — different from Istanbul's HNR05 (0-500Hz band)

### 3.2 Domains B, C, D — Istanbul Replicated Acoustic Features
- **Source:** UCI ML Repository #489, Naranjo et al. (2016)
- **URL:** https://archive.ics.uci.edu/dataset/489
- **Size:** 240 total = 80 subjects × 3 recording sessions
  - Domain B = Session R1: 80 samples (40 PD + 40 Healthy)
  - Domain C = Session R2: 80 samples (40 PD + 40 Healthy)
  - Domain D = Session R3: 80 samples (40 PD + 40 Healthy)
- **Class balance:** Exactly 50/50 PD/Healthy — **perfectly balanced**
- **Recording protocol:** Sustained phonation, replicated from Oxford protocol
  but different equipment, different lab (Bogazici University, Turkey)
- **Features:** 12 shared features (same names) + 35 extended:
  - Extended: MFCC 0–12 (13), Delta 0–12 (13), GNE, HNR15, HNR25, HNR35, HNR38 — total ~47
  - **Current implementation uses only 12 shared features for cross-domain experiments**
- **Patient split:** Not patient-wise (each patient appears once per session),
  random 80/20 split used for in-domain experiments
- **Known issues:**
  - Only 80 samples per session (very small)
  - Same 80 patients across B/C/D → B,C,D are NOT truly independent domains;
    they share all patients (only the recording session differs)
  - 3 sessions replicate the same phonation → very high between-session correlation
  - HNR05 ≠ MDVP:HNR → systematic feature mismatch with Oxford

### 3.3 Cross-domain Feature Alignment Issue
The 12 "shared" features between Oxford and Istanbul are **nominally the same**
but measured with different software/hardware:
- `jitter_rel`: Oxford uses MDVP protocol; Istanbul uses Praat — different normalization
- `hnr`: Oxford = MDVP broadband; Istanbul = HNR05 (0–500 Hz only)
- `shimmer_db`: different reference levels possible

This **intentional domain shift** is the core challenge of the task, but it also
means the feature distributions are structurally different even for the same
underlying pathology. This is a realistic clinical scenario.

---

## 4. Model Architecture Details

### 4.1 Feature Extractor (shared in all DA models)
```
Input(12) → Linear(12→128) → BatchNorm1d → ReLU → Dropout(0.3)
           → Linear(128→64) → BatchNorm1d → ReLU
```
Feature dimension: 64

### 4.2 Label Predictor (classification head)
```
Linear(64→2) → CrossEntropyLoss
```

### 4.3 Domain heads (method-specific)
- **DANN/CDAN:** `Linear(64→32) → ReLU → Dropout → Linear(32→num_domains)`
- **CDAN specific:** domain classifier input = `feature ⊗ softmax(class_logits)` → dim 64×2=128
- **MMD/CORAL:** no domain head; alignment via loss term on feature distributions
- **Contrastive:** prototype memory per class; NT-Xent loss with τ=0.07

### 4.4 Training hyperparameters
| Parameter | Value |
|---|---|
| Batch size (standard) | 32 |
| Batch size (DA) | 64 |
| LR (standard) | 0.001 |
| LR (DA) | 0.0005 |
| Epochs (standard) | 50 |
| Epochs (DA) | 100 |
| Optimizer | Adam + weight_decay=1e-4 |
| Grad clip | 1.0 |
| Dropout | 0.3 |
| DANN λ schedule | sigmoid: 2/(1+exp(-10p))−1, saturates ≈0.5 |
| LR schedule (DANN) | inverse decay: 1/(1+10·p)^0.75 |
| CORAL λ | 1.0 |
| MMD λ | 0.5 |
| CDAN λ | 0.5 |
| Contrastive λ | 0.5, τ=0.07, confidence threshold=0.75 |

---

## 5. Experimental Results (Last Successful Run, ~27s CPU)

### 5.1 In-domain results (upper bound — same distribution train/test)

| Method | Domain | Acc | F1 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| MLP | A (Oxford) | 0.796 | 0.881 | 0.910 | 1.000 | 0.167 |
| CNN1D | A (Oxford) | 0.816 | 0.892 | 0.971 | 1.000 | 0.250 |
| SVM | A (Oxford) | 0.571 | 0.687 | 0.511 | 0.622 | 0.417 |
| MLP | B (Istanbul R1) | 0.688 | 0.667 | 0.719 | 0.625 | 0.750 |
| SVM | B (Istanbul R1) | 0.688 | 0.667 | 0.734 | 0.625 | 0.750 |
| MLP | D (Istanbul R3) | 0.625 | 0.571 | 0.625 | 0.500 | 0.750 |

### 5.2 Cross-domain baseline (no adaptation — quantifies domain gap)

| Method | Direction | Acc | F1 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| MLP | A→B | 0.487 | 0.586 | 0.579 | 0.725 | 0.250 |
| SVM | A→B | 0.475 | 0.543 | 0.523 | 0.625 | 0.325 |
| MLP | B→A | 0.569 | 0.635 | 0.728 | 0.497 | 0.792 |

### 5.3 Domain Adaptation A→B (primary experiment)

| Method | Acc | F1 | AUC | Sensitivity | Specificity | ΔAUC vs baseline |
|---|---|---|---|---|---|---|
| MLP baseline | 0.487 | 0.586 | 0.579 | 0.725 | 0.250 | — |
| DANN | 0.475 | 0.571 | 0.602 | 0.700 | 0.250 | +0.023 |
| MMD | 0.525 | 0.620 | 0.586 | 0.775 | 0.275 | +0.007 |
| CORAL | 0.525 | 0.620 | 0.571 | 0.775 | 0.275 | −0.008 |
| CDAN | 0.487 | 0.594 | 0.580 | 0.750 | 0.225 | +0.001 |
| Contrastive | 0.550 | 0.667 | 0.614 | 0.900 | 0.200 | +0.035 |

### 5.4 Multi-source DA: (B+C)→D

| Method | Acc | F1 | AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Multi-DANN | 0.750 | 0.730 | 0.834 | 0.675 | 0.825 |
| Multi-CORAL | 0.750 | 0.737 | 0.853 | 0.700 | 0.800 |

### 5.5 Bootstrap 95% Confidence Intervals on AUC (A→B, N=1000)

| Method | AUC | 95% CI lower | 95% CI upper | CI width |
|---|---|---|---|---|
| MLP baseline | 0.579 | 0.452 | 0.712 | 0.260 |
| DANN | 0.602 | 0.470 | 0.734 | 0.264 |
| MMD | 0.586 | 0.452 | 0.717 | 0.265 |
| CORAL | 0.571 | 0.436 | 0.701 | 0.265 |
| CDAN | 0.580 | 0.451 | 0.712 | 0.261 |
| Contrastive | 0.614 | 0.479 | 0.736 | 0.257 |

---

## 6. Critical Observations on Results

### 6.1 Statistical significance
All 95% CI intervals overlap completely across all methods on A→B.
The CI width (~0.26) is enormous relative to the AUC differences (~0.02–0.04).
**No method is statistically distinguishable from the baseline on A→B.**
This is a fundamental validity concern for the experiment.

### 6.2 AUC values in context
- A→B AUC range: 0.571–0.614 — only marginally above random (0.5)
- In-domain CNN A: AUC 0.971 — the model CAN learn the task
- The gap (0.971 → 0.579) represents extreme domain shift
- Multi-source (B+C)→D: AUC 0.853 — dramatically better; but B,C,D share patients
  (same 80 patients across sessions) making this a weaker test of generalization

### 6.3 Accuracy as metric — known issue
Oxford test set: ~48 samples, 75% PD. A model predicting ALL samples as PD achieves
accuracy ≈ 0.75, F1 ≈ 0.857 automatically. The Oxford in-domain MLP (sensitivity=1.000,
specificity=0.167) appears to exhibit this behavior: it classifies essentially
everything as PD. The AUC=0.910 is more reliable here but still suspect given
the degenerate behavior indicated by specificity=0.167.

### 6.4 Class imbalance handling
Oxford: 75% PD. `compute_class_weights()` is implemented in datasets.py but
**class_weights are NOT passed to train_model() or DA trainers** in main.py.
The default training uses unweighted CrossEntropyLoss throughout.

### 6.5 Domain shift source analysis
The A→B experiment (Oxford→Istanbul) contains multiple simultaneous shifts:
1. **Lab shift:** different recording environment
2. **Equipment shift:** MDVP vs Praat software
3. **Population shift:** Oxford (UK) vs Istanbul (Turkey) — different PD demographics
4. **Protocol shift:** single recording vs 3 sessions (only R1 used for B)
5. **Feature scale shift:** jitter_rel different normalization; HNR different bandwidth
6. **Class prior shift:** Oxford 75% PD vs Istanbul 50% PD

### 6.6 Multi-source inflated results
The multi-source experiment (B+C)→D achieves AUC 0.853.
However, B,C,D are all from the same 80 Istanbul patients (just different recording sessions).
Training on sessions 1+2 and testing on session 3 of the **same patients** is closer
to within-subject cross-session validation than true cross-domain adaptation.
This inflates the apparent generalization.

### 6.7 Test set sizes
- A test: ~49 samples (25% of 195, patient-wise) — very small, high variance estimates
- B test: 16 samples (20% of 80) — extremely small; each prediction changes metrics by ~6%
- D test: 16 samples — same issue

---

## 7. Dataset Context vs. Assignment Requirements

The assignment explicitly names **PDITA, Neurovoz, PC-GITA** as example datasets.
The current implementation uses **Oxford UCI and Istanbul UCI** instead.

Key differences of the suggested datasets:
- **PC-GITA:** 50 PD + 50 Healthy Spanish speakers; continuous speech + sustained
  phonation; multiple microphones; widely used benchmark for DA in PD detection
- **Neurovoz:** Spanish, 32 PD + 32 Healthy; multiple tasks including sustained
  phonation, DDK, connected speech
- **PDITA:** Italian dataset, sustained phonation

These datasets provide **raw audio** enabling:
- Spectrogram-based CNN (as required by the assignment)
- Foundation model feature extraction (wav2vec2, HuBERT, Whisper)
- More diverse phonation tasks beyond sustained /a/
- Better geographic/linguistic diversity for domain shift

The Oxford and Istanbul datasets only provide **pre-extracted scalar features**,
making it impossible to implement the spectrogram CNN or foundation model classifiers
listed in the assignment's classifier options.

---

## 8. Code Quality Notes

### 8.1 Strengths
- Patient-wise split correctly implemented (no data leakage)
- BatchNorm in feature extractor aids domain adaptation
- DANN λ schedule follows the original paper (sigmoid schedule)
- DANN LR schedule follows original paper (inverse decay)
- Multi-kernel MMD with median heuristic bandwidth
- Bootstrap CI correctly implemented (stratified by sample, not by method)
- All plots saved to files (no GUI dependency)

### 8.2 Potential Issues
- `class_weights` not used in training despite being computed
- CORAL λ=1.0 is large; no sensitivity analysis on λ values
- Contrastive confidence threshold 0.75 with only 80 target samples may leave
  too few pseudo-labeled samples per class (~20-25 per epoch)
- Target data IS available at test time but pseudo-labels used during training
  may be noisy (initial classifier trained only on source)
- No early stopping — fixed 100 epochs regardless of convergence
- No cross-validation — single train/test split means high variance results
- CDAN uses `feature ⊗ softmax` directly without the random multilinear map
  from the original paper (approximation)

---

## 9. Project File Structure
```
parkinson_da/
├── main.py                        # 447 lines, 10 experiment sections
├── requirements.txt               # torch, pandas, numpy, sklearn, matplotlib
├── src/
│   ├── config.py                  # All hyperparameters centralized
│   ├── utils.py                   # ROC, t-SNE, loss, performance drop plots
│   ├── data/
│   │   ├── download_data.py       # Downloads Oxford + Istanbul from UCI
│   │   └── datasets.py            # DataLoaders, patient-wise split, scaler
│   ├── models/
│   │   ├── models.py              # MLP, CNN1D
│   │   └── domain_adaptation.py   # DANN, MMD, CORAL, CDAN, Contrastive (349 lines)
│   ├── training/
│   │   └── train.py               # train_model, train_dann, train_mmd,
│   │                              # train_coral, train_cdan, train_contrastive,
│   │                              # train_multisource_dann
│   └── evaluation/
│       └── evaluate.py            # evaluate_model, evaluate_svm, get_roc_data,
│                                  # extract_features, get_predictions, bootstrap_ci
└── data/                          # Generated on first run
    ├── oxford.csv / istanbul_r*.csv
    ├── roc_curves_AtoB.png
    ├── tsne_dann.png
    ├── performance_drop.png
    └── losses_da.png
```

---

## 10. Environment
- Python 3.12.2
- PyTorch 2.11.0 (CPU only, no CUDA detected)
- scikit-learn, matplotlib, pandas, numpy
- venv at `.venv/`
- Run command: `python main.py` (data downloads automatically on first run)
- Runtime: ~27 seconds on CPU
