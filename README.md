# Detekcia Parkinsonovej choroby z reči - Domain Adaptation

Semestrálny projekt INS: Porovnanie domain adaptation techník
pre detekciu Parkinsonovej choroby z hlasových nahrávok.

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

| Technika | Popis                             | Referencia                                                           |
| -------- | --------------------------------- | -------------------------------------------------------------------- |
| Baseline | Bez adaptacie (train A → test B)  | -                                                                    |
| DANN     | Domain-Adversarial Neural Network | [Ganin et al., 2015](https://arxiv.org/abs/1409.7495)               |
| MMD      | Maximum Mean Discrepancy          | [Gretton et al., 2012](https://jmlr.org/papers/v13/gretton12a.html) |
| CORAL    | CORrelation ALignment (TODO)      | [Sun & Saenko, 2016](https://arxiv.org/abs/1607.01719)              |

## Inštalácia a spustenie

```bash
# 1. Inštalácia závislostí
pip install -r requirements.txt

# 2. Spustenie experimentu
python main.py

# Alebo po krokoch:
python download_data.py    # len stiahnutie dát
python main.py             # trénovanie + evaluácia
```

## Štruktúra projektu

```text
parkinson_da/
├── config.py              # Konfigurácia a hyperparametre
├── download_data.py       # Stiahnutie datasetov z UCI
├── datasets.py            # PyTorch Dataset triedy
├── models.py              # MLP a CNN klasifikátory
├── domain_adaptation.py   # DANN, MMD, GRL
├── train.py               # Trénovacie funkcie
├── evaluate.py            # Evaluačné metriky
├── utils.py               # Vizualizácia a utility
├── main.py                # Hlavný skript
├── requirements.txt       # Závislosti
├── README.md              # Dokumentácia
└── data/                  # Stiahnuté dáta (generované)
    ├── oxford.csv
    └── istanbul.csv
```

## Týždenný plán

### Implementované (Týždeň 1-8)

| Tyzden | Datum          | Uloha                                               | Cvicenie              |
| ------ | -------------- | --------------------------------------------------- | --------------------- |
| 1      | 16-22 Feb      | Setup projektu, download datasetov                  | Cv.1 - Tensory        |
| 2      | 23 Feb - 1 Mar | Dataset triedy, DataLoader, exploracia dat           | Cv.2 - Praca s datami |
| 3      | 2-8 Mar        | Standardizacia priznakov, nn.Module architektura     | Cv.3 - Komponenty NN  |
| 4      | 9-15 Mar       | MLP klasifikator + trenovaci cyklus                 | Cv.4 - FF siete       |
| 5      | 16-22 Mar      | CNN1D klasifikator                                  | Cv.5 - CNN            |
| 6      | 23-28 Mar      | SVM baseline + in-domain evaluacia (F1, AUC)        | Cv.6 - LSTM           |
| 7      | 30 Mar - 5 Apr | DANN implementacia (GRL + adversarialne trenovanie) | Projekt               |
| 8      | 6-12 Apr       | MMD implementacia + cross-domain baseline            | Projekt               |

### TODO (Týždeň 9-13)

| Tyzden | Datum           | Uloha                                              |
| ------ | --------------- | -------------------------------------------------- |
| 9      | 13-19 Apr       | CORAL implementacia (Deep CORAL)                   |
| 10     | 20-26 Apr       | Multi-source domain adaptation                     |
| 11     | 27 Apr - 3 May  | Kompletna evaluacia (oba smery, statistika)        |
| 12     | 4-10 May        | Vizualizacia vysledkov, ROC krivky, t-SNE          |
| 13     | 11-17 May       | Zaverecna sprava a prezentacia                     |

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

5. Sun, B. & Saenko, K. (2016). "Deep CORAL: Correlation Alignment for
   Deep Domain Adaptation." ECCV 2016. <https://arxiv.org/abs/1607.01719>
