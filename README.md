# Porovnanie Domain Adaptation Techník pre Detekciu Parkinsonovej Choroby z Reči

## Autori
- **Dmytro Protsun** (dmytro.protsun@student.tuke.sk)
- **Mykyta Olym** (mykyta.olym@student.tuke.sk)

**Predmet:** INS
**Semestrálny projekt 2025/2026**  
**TUKE – Technická univerzita v Košiciach**

---

## Popis projektu

Tento projekt implementuje systém na detekciu Parkinsonovej choroby (PD) z hlasových nahrávok. Porovnávame rôzne klasifikátory a domain adaptation techniky na troch verejných databázach.

### Klasifikátory
1. **CNN model** – konvolučná neurónová sieť nad mel-spektrogramami
2. **Tradičný model (MLP/SVM)** – nad extrahovanými akustickými príznakmi (MFCC, jitter, shimmer, HNR)
3. **Wav2Vec2** – foundation model pre reč (predtrénovaný na veľkom množstve audio dát)

### Databázy (domény)
- **PC-GITA** – kolumbijský dataset (španielčina)
- **Neurovoz** – španielsky dataset z Madridu
- **PDITA** – taliansky dataset

### Domain Adaptation techniky
1. **Baseline** – bez adaptácie (tréning na source, test na target)
2. **DANN** – Domain-Adversarial Neural Network (gradient reversal)
3. **MMD** – Maximum Mean Discrepancy (moment matching)
4. **Contrastive** – Contrastive Domain Alignment
5. **Multi-source** – Multi-Source Domain Adaptation

### Metriky
- Accuracy, AUC (Area Under ROC Curve), F1-score
- Senzitivita (recall pre PD), Špecificita (recall pre Healthy)

---

## Štruktúra projektu

```
INS/
├── main.py                          # Hlavný vstupný bod
├── requirements.txt                 # Závislosti
├── README.md                        # Tento súbor
│
├── config/                          # Konfigurácia
│   ├── __init__.py
│   └── settings.py                  # Všetky nastavenia a hyperparametre
│
├── data/                            # Načítanie dát
│   ├── __init__.py
│   ├── base_dataset.py              # Základná trieda pre datasety
│   ├── pcgita_dataset.py            # PC-GITA dataset loader
│   ├── neurovoz_dataset.py          # Neurovoz dataset loader
│   ├── pdita_dataset.py             # PDITA dataset loader
│   └── data_loader.py               # DataLoader factory + splits
│
├── preprocessing/                   # Predspracovanie audia
│   ├── __init__.py
│   ├── audio_utils.py               # Základné audio operácie
│   ├── feature_extraction.py        # Extrakcia akustických príznakov
│   └── spectrogram_maker.py         # Vytváranie spektrogramov
│
├── models/                          # Architektúry modelov
│   ├── __init__.py
│   ├── cnn_classifier.py            # CNN na spektrogramoch
│   ├── traditional_classifier.py    # SVM a MLP klasifikátory
│   ├── wav2vec_classifier.py        # Wav2Vec2 klasifikátor
│   └── feature_extractor_net.py     # Spoločný feature extractor
│
├── domain_adaptation/               # DA techniky
│   ├── __init__.py
│   ├── baseline.py                  # Bez adaptácie
│   ├── dann.py                      # DANN (gradient reversal)
│   ├── mmd_adaptation.py            # MMD (moment matching)
│   ├── contrastive_alignment.py     # Contrastive alignment
│   └── multi_source_adaptation.py   # Multi-source DA
│
├── training/                        # Tréning a evaluácia
│   ├── __init__.py
│   ├── trainer.py                   # Hlavný tréner
│   └── evaluator.py                 # Evaluácia s metrikami
│
├── utils/                           # Pomocné funkcie
│   ├── __init__.py
│   ├── metrics.py                   # Klasifikačné metriky
│   ├── visualization.py             # Grafy a vizualizácie
│   └── helpers.py                   # Rôzne utility
│
├── experiments/                     # Spúšťanie experimentov
│   ├── __init__.py
│   ├── run_all_experiments.py       # Spustenie všetkých experimentov
│   └── compare_results.py           # Porovnanie výsledkov
│
├── datasets/                        # TU ULOŽTE DATASETY
│   ├── PC-GITA/
│   │   ├── healthy/
│   │   └── parkinsons/
│   ├── Neurovoz/
│   │   ├── healthy/
│   │   └── parkinsons/
│   └── PDITA/
│       ├── healthy/
│       └── parkinsons/
│
├── results/                         # Výsledky (vytvorí sa automaticky)
│   ├── all_results.csv
│   ├── evaluation_report.txt
│   ├── latex_table.tex
│   └── plots/
│
└── saved_models/                    # Uložené modely (vytvorí sa automaticky)
```

---

## Inštalácia

### 1. Klonovanie repozitára
```bash
git clone <url-repozitara>
cd INS
```

### 2. Vytvorenie virtuálneho prostredia (odporúčané)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Inštalácia závislostí
```bash
pip install -r requirements.txt
```

### 4. Príprava datasetov
Stiahnite datasety a umiestnite ich do priečinka `datasets/`:

- **PC-GITA**: Kontaktujte autorov datasetu
- **Neurovoz**: https://zenodo.org/record/2867216
- **PDITA**: Kontaktujte autorov datasetu

Očakávaná štruktúra:
```
datasets/
├── PC-GITA/
│   ├── healthy/     # WAV súbory zdravých kontrol
│   └── parkinsons/  # WAV súbory PD pacientov
├── Neurovoz/
│   ├── healthy/
│   └── parkinsons/
└── PDITA/
    ├── healthy/
    └── parkinsons/
```

---

## Spustenie

### Jeden experiment
```bash
# CNN + Baseline: PC-GITA -> Neurovoz
python main.py --model cnn --method baseline --source PC-GITA --target Neurovoz

# CNN + DANN: PC-GITA -> PDITA
python main.py --model cnn --method dann --source PC-GITA --target PDITA --epochs 30

# MLP + MMD: Neurovoz -> PC-GITA
python main.py --model traditional --method mmd --source Neurovoz --target PC-GITA

# Wav2Vec2 + Contrastive
python main.py --model wav2vec --method contrastive --source PC-GITA --target Neurovoz

# Multi-source: PC-GITA + Neurovoz -> PDITA
python main.py --model cnn --method multi_source --target PDITA
```

### Všetky experimenty
```bash
# Spustí všetky kombinácie (trvá dlho!)
python main.py --run-all --epochs 50

# Rýchly test (overenie funkčnosti)
python main.py --quick
```

### Všetky DA metódy pre jeden model
```bash
python main.py --model cnn --method all --source PC-GITA --target Neurovoz
```

---

## Výsledky

Po spustení experimentov sa výsledky uložia do priečinka `results/`:

- `all_results.csv` – všetky metriky v CSV formáte
- `evaluation_report.txt` – textová správa
- `latex_table.tex` – LaTeX tabuľka (pre seminárnu prácu)
- `plots/` – grafy (confusion matrix, ROC krivky, porovnania)

---

## Technické detaily

### Domain Adaptation – princípy

**Baseline (bez adaptácie):**
- Natrénujeme model na source doméne
- Otestujeme na target doméne
- Očakávame pokles výkonu kvôli domain shift

**DANN (Domain-Adversarial Neural Network):**
- Gradient Reversal Layer medzi feature extractorom a domain classifierom
- Feature extractor sa učí domain-invariantné features
- Lambda parameter sa postupne zvyšuje (schedule)

**MMD (Maximum Mean Discrepancy):**
- Meria štatistickú vzdialenosť medzi distribúciami features
- Multi-kernel prístup s viacerými bandwidth hodnotami
- Minimalizáciou MMD zbližujeme source a target features

**Contrastive Domain Alignment:**
- Supervised contrastive loss na source dátach
- Cross-domain contrastive alignment
- Projection head mapuje features do normalizovaného priestoru

**Multi-Source Domain Adaptation:**
- Kombinácia viacerých source domén
- Vážené príspevky podľa MMD vzdialenosti
- Viac trénovacích dát = lepšia generalizácia

### Akustické príznaky
- **MFCC** – Mel-Frequency Cepstral Coefficients (13 koeficientov + delta + delta-delta)
- **Jitter** – variabilita základnej frekvencie (F0)
- **Shimmer** – variabilita amplitúdy
- **HNR** – Harmonics-to-Noise Ratio
- **Spektrálne príznaky** – spectral centroid, bandwidth, rolloff, ZCR

---

## Referencie

1. Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR.
2. Long, M. et al. (2015). "Learning Transferable Features with Deep Adaptation Networks." ICML.
3. Khosla, P. et al. (2020). "Supervised Contrastive Learning." NeurIPS.
4. Peng, X. et al. (2019). "Moment Matching for Multi-Source Domain Adaptation." ICCV.
5. Orozco-Arroyave, J.R. et al. (2014). "New Spanish speech corpus for PD detection." (PC-GITA)
6. Neurovoz corpus: https://zenodo.org/record/2867216

---

## Licencia

Tento projekt je vytvorený pre akademické účely v rámci semestrálneho projektu na TUKE.
