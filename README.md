# Porovnanie Domain Adaptation Techník pre Detekciu Parkinsonovej Choroby z Reči

## Autori

- **Dmytro Protsun** (<dmytro.protsun@student.tuke.sk>)
- **Mykyta Olym** (<mykyta.olym@student.tuke.sk>)

**Predmet:** INS
**Semestrálny projekt 2025/2026**
**TUKE – Technická univerzita v Košiciach**

---

## Popis projektu

Tento projekt implementuje systém na detekciu Parkinsonovej choroby (PD) z hlasových nahrávok. Porovnávame rôzne klasifikátory a domain adaptation techniky na dvoch verejne dostupných databázach.

### Klasifikátory

1. **CNN model** – konvolučná neurónová sieť nad mel-spektrogramami
2. **Tradičný model (MLP)** – nad extrahovanými akustickými príznakmi (MFCC, jitter, shimmer, HNR)
3. **Wav2Vec2** – foundation model pre reč (predtrénovaný na veľkom množstve audio dát)

### Databázy (domény)

- **MDVR-KCL** – anglický dataset z King's College London (smartfón nahrávky, 36 účastníkov)
- **ItalianPVS** – taliansky dataset z Università degli Studi di Bari (65 účastníkov)

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

```text
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
│   ├── mdvr_kcl_dataset.py          # MDVR-KCL dataset loader
│   ├── italian_pvs_dataset.py       # ItalianPVS dataset loader
│   └── data_loader.py               # DataLoader factory + splits
│
├── preprocessing/                   # Predspracovanie audia
│   ├── __init__.py
│   ├── feature_extraction.py        # Extrakcia akustických príznakov
│   └── spectrogram_maker.py         # Vytváranie spektrogramov
│
├── models/                          # Architektúry modelov
│   ├── __init__.py
│   ├── cnn_classifier.py            # CNN na spektrogramoch
│   ├── traditional_classifier.py    # MLP klasifikátor
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
│   ├── MDVR-KCL/                   # MDVR-KCL dataset (z Zenodo)
│   │   └── 26-29_09_2017_KCL/
│   │       ├── ReadText/{HC,PD}/
│   │       └── SpontaneousDialogue/{HC,PD}/
│   └── ItalianPVS/                  # ItalianPVS dataset (z IEEE DataPort)
│       ├── HC/ alebo Young_HC/ + Elderly_HC/
│       └── PD/
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

#### MDVR-KCL (Open Access)

- **Zdroj**: <https://zenodo.org/record/2867216>
- **Licencia**: CC-BY 4.0 (plne otvorený)
- **Jazyk**: Angličtina
- **Účastníci**: 21 HC + 15 PD = 36 celkom
- **Úlohy**: ReadText (čítanie textu "The North Wind and the Sun"), SpontaneousDialogue (spontánny dialóg)
- **Zariadenie**: Motorola Moto G4 smartphone, 44.1 kHz
- **Veľkosť**: 606 MB
- Stiahnite `26_29_09_2017_KCL.zip`, rozbaľte a obsah umiestnite do `datasets/MDVR-KCL/`

```text
datasets/MDVR-KCL/
  └── 26-29_09_2017_KCL/
      ├── ReadText/
      │   ├── HC/           # ID00_hc_0_0_0.wav, ID01_hc_0_0_0.wav, ...
      │   └── PD/           # ID02_pd_2_0_0.wav, ID04_pd_2_0_1.wav, ...
      └── SpontaneousDialogue/
          ├── HC/
          └── PD/
```

Schéma pomenovania: `ID{NN}_{hc|pd}_{H&Y}_{UPDRS_II-5}_{UPDRS_III-18}.wav`

#### ItalianPVS (Open Access – bezplatný IEEE účet)

- **Zdroj**: <https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech>
- **DOI**: 10.21227/aw6b-tg17
- **Licencia**: Open Access (vyžaduje sa bezplatný IEEE účet na stiahnutie)
- **Jazyk**: Taliančina
- **Účastníci**: 15 Young HC + 22 Elderly HC + 28 PD = 65 celkom
- **Formát**: WAV audio súbory + XLSX metadata
- **Veľkosť**: 565 MB
- Vytvorte si bezplatný IEEE účet, stiahnite ZIP a rozbaľte do `datasets/ItalianPVS/`

```text
datasets/ItalianPVS/
  ├── metadata.xlsx          # (ak je súčasťou)
  ├── Young_HC/              # 15 mladých zdravých kontrol
  │   └── *.wav
  ├── Elderly_HC/            # 22 starších zdravých kontrol
  │   └── *.wav
  └── PD/                    # 28 pacientov s Parkinsonovou chorobou
      └── *.wav
```

**Poznámka:** Loader automaticky detekuje rôzne štruktúry priečinkov (HC/PD, healthy/parkinsons, atď.)

**Citácia:**

> G. Dimauro, V. Di Nicola, V. Bevilacqua, D. Caivano and F. Girardi,
> "Assessment of Speech Intelligibility in Parkinson's Disease Using a
> Speech-To-Text System," IEEE Access, vol. 5, pp. 22199-22208, 2017.
> doi: 10.1109/ACCESS.2017.2762475.

---

## Spustenie

### Jeden experiment

```bash
# CNN + Baseline: MDVR-KCL -> ItalianPVS
python main.py --model cnn --method baseline --source MDVR-KCL --target ItalianPVS

# CNN + DANN: ItalianPVS -> MDVR-KCL
python main.py --model cnn --method dann --source ItalianPVS --target MDVR-KCL --epochs 30

# MLP + MMD: ItalianPVS -> MDVR-KCL
python main.py --model traditional --method mmd --source ItalianPVS --target MDVR-KCL

# Wav2Vec2 + Contrastive
python main.py --model wav2vec --method contrastive --source MDVR-KCL --target ItalianPVS

# Multi-source: obe domény
python main.py --model cnn --method multi_source --target ItalianPVS
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
python main.py --model cnn --method all --source MDVR-KCL --target ItalianPVS
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

### Domain Shift medzi datasetmi

| Vlastnosť | MDVR-KCL | ItalianPVS |
| --- | --- | --- |
| Jazyk | Angličtina | Taliančina |
| Krajina | UK (Londýn) | Taliansko (Bari) |
| Účastníci | 21 HC + 15 PD | 37 HC + 28 PD |
| Zariadenie | Smartphone (44.1 kHz) | Mikrofón |
| Úlohy | Čítanie textu, Dialóg | Čítanie textu, Spontánna reč |

Tieto rozdiely vytvárajú **domain shift**, čo je presne to, čo domain adaptation techniky riešia.

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
5. Jaeger, H. et al. (2019). "Mobile Device Voice Recordings at King's College London (MDVR-KCL)." Zenodo. <https://doi.org/10.5281/zenodo.2867216>
6. Dimauro, G. et al. (2017). "Assessment of Speech Intelligibility in Parkinson's Disease Using a Speech-To-Text System." IEEE Access. doi: 10.1109/ACCESS.2017.2762475

---

## Licencia

Tento projekt je vytvorený pre akademické účely v rámci semestrálneho projektu na TUKE.
