# Detekcia Parkinsonovej choroby z reči — Domain Adaptation

Semestrálny projekt INS: Porovnanie domain adaptation techník
pre detekciu Parkinsonovej choroby z hlasových nahrávok.

## Rýchly štart

```bash
# 1. Klonovanie repozitára
git clone <URL repozitára>
cd parkinson_da

# 2. Vytvorenie virtuálneho prostredia
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Inštalácia závislostí
pip install -r requirements.txt

# 4. Spustenie
python main.py
```

> **Dáta sa stiahnu automaticky** z UCI ML Repository pri prvom spustení (~1 MB).
> Výsledky a grafy sa uložia do priečinka `data/`.

---

## Požiadavky

- Python **3.10+**
- ~300 MB miesta na disku (venv + závislosti)
- Internetové pripojenie pri prvom spustení (stiahnutie dát)
- CPU postačuje — tréning beží cca 30 sekúnd

---

## Popis

Projekt porovnáva rôzne prístupy k domain adaptation pre klasifikáciu
Parkinson's Disease (PD) vs. Healthy na základe akustických príznakov
extrahovaných z hlasových nahrávok.

## Datasety (domény)

Oba datasety sú voľne dostupné z UCI ML Repository (CC BY 4.0):

| Domena | Dataset                                              | Vzorky | Zdroj                                                            |
| ------ | ---------------------------------------------------- | ------ | ---------------------------------------------------------------- |
| A      | Oxford Parkinson's (Little et al., 2008)             | 195    | [UCI #174](https://archive.ics.uci.edu/dataset/174/parkinsons)   |
| B      | Istanbul Replicated Features (Naranjo et al., 2016)  | 240    | [UCI #489](https://archive.ics.uci.edu/dataset/489)              |

**12 spoločných akustických príznakov**: Jitter (4x), Shimmer (4x), HNR, RPDE, DFA, PPE

## Klasifikátory

| Model | Popis                                | Cvicenie         |
| ----- | ------------------------------------ | ---------------- |
| MLP   | Feed-forward siet (12→64→32→2)       | Cvicenie 4 - FF  |
| CNN1D | 1D konvolucna siet                   | Cvicenie 5 - CNN |
| SVM   | Support Vector Machine (scikit-learn) | baseline         |

## Domain Adaptation techniky

| Technika    | Popis                                          | Referencia                                                            |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------------- |
| Baseline    | Bez adaptácie (train A → test B)               | —                                                                     |
| DANN        | Domain-Adversarial Neural Network              | [Ganin et al., 2015](https://arxiv.org/abs/1409.7495)                |
| MMD         | Maximum Mean Discrepancy                       | [Gretton et al., 2012](https://jmlr.org/papers/v13/gretton12a.html)  |
| CORAL       | CORrelation ALignment (Deep CORAL)             | [Sun & Saenko, 2016](https://arxiv.org/abs/1607.01719)               |
| CDAN        | Conditional DA with multilinear conditioning   | [Long et al., 2018](https://arxiv.org/abs/1705.10667)                |
| Contrastive | Prototype-based contrastive DA (NT-Xent)       | [Yang et al., 2021](https://arxiv.org/abs/2101.09209)                |

## Štruktúra projektu

```text
parkinson_da/
├── main.py                      # Hlavný skript — spustí všetky experimenty
├── requirements.txt
├── src/
│   ├── config.py                # Hyperparametre a konštanty
│   ├── utils.py                 # Vizualizácia (ROC, t-SNE, grafy strát)
│   ├── data/
│   │   ├── download_data.py     # Stiahnutie datasetov z UCI
│   │   └── datasets.py          # DataLoader, patient-wise split
│   ├── models/
│   │   ├── models.py            # MLP, CNN1D
│   │   └── domain_adaptation.py # DANN, MMD, CORAL, CDAN, Contrastive
│   ├── training/
│   │   └── train.py             # Trénovacie funkcie pre všetky metódy
│   └── evaluation/
│       └── evaluate.py          # Metriky, ROC, bootstrap CI
└── data/                        # Generované pri spustení
    ├── oxford.csv
    ├── istanbul_r1/r2/r3.csv
    ├── roc_curves_AtoB.png
    ├── tsne_dann.png
    ├── performance_drop.png
    └── losses_da.png
```

## Výsledky

### In-domain (horná hranica)

| Model | Doména | AUC   | F1    | Acc   |
|-------|--------|-------|-------|-------|
| CNN1D | A (Oxford) | **0.971** | 0.892 | 0.816 |
| MLP   | A (Oxford) | 0.910 | 0.881 | 0.796 |
| MLP   | B (Istanbul R1) | 0.719 | 0.667 | 0.688 |

### Cross-domain A→B (domain gap)

| Metóda       | AUC       | F1    | Acc   |
|--------------|-----------|-------|-------|
| MLP baseline | 0.579     | 0.586 | 0.487 |
| DANN         | 0.602     | 0.571 | 0.475 |
| MMD          | 0.586     | 0.620 | 0.525 |
| CORAL        | 0.571     | 0.620 | 0.525 |
| CDAN         | 0.580     | 0.594 | 0.487 |
| **Contrastive** | **0.614** | **0.667** | **0.550** |

### Multi-source DA: (B+C)→D

| Metóda       | AUC       | F1    | Acc   |
|--------------|-----------|-------|-------|
| Multi-DANN   | 0.834     | 0.730 | 0.750 |
| **Multi-CORAL** | **0.853** | **0.737** | **0.750** |

## Metriky

- **Accuracy** - celková presnosť
- **F1 skóre** - harmonický priemer presnosti a úplnosti
- **AUC** - plocha pod ROC krivkou
- **Senzitivita** - schopnosť zachytiť PD pacientov (TP / (TP+FN))
- **Špecificita** - schopnosť identifikovať zdravých (TN / (TN+FP))

## Referencie

1. Little, M.A. et al. (2008). "Suitability of dysphonia measurements for
   telemonitoring of Parkinson's disease." IEEE Trans. Biomedical Engineering.

2. Naranjo, L. et al. (2016). "Addressing voice recording replications for
   Parkinson's disease detection." Expert Systems with Applications.

3. Ganin, Y. & Lempitsky, V. (2015). "Unsupervised Domain Adaptation by
   Backpropagation." ICML 2015. <https://arxiv.org/abs/1409.7495>

4. Gretton, A. et al. (2012). "A Kernel Two-Sample Test."
   JMLR 13. <https://jmlr.org/papers/v13/gretton12a.html>

4. Long, M. et al. (2018). "Conditional Adversarial Domain Adaptation."
   NeurIPS 2018. <https://arxiv.org/abs/1705.10667>

5. Yang, L. et al. (2021). "Exploiting the Intrinsic Neighborhood Structure
   for Source-free Domain Adaptation." NeurIPS 2021. <https://arxiv.org/abs/2110.04202>

6. Sun, B. & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for
   Deep Domain Adaptation." ECCV 2016. <https://arxiv.org/abs/1607.01719>
