# INS — Inteligentné Neurónové Siete (TUKE)

Repozitár pre predmet **Inteligentné neurónové siete** — cvičenia a semestrálny projekt.

## Štruktúra

```
├── Cviki/                 # Cvičenia 1–7
│   ├── 01_Tensory/        # PyTorch tensory, operácie
│   ├── 02_Praca_s_datami/ # Dataset, DataLoader, transformácie
│   ├── 03_Komponenty_NN/  # nn.Module, vrstvy, aktivácie
│   ├── 04_FeedForward/    # MLP klasifikátor (MNIST)
│   ├── 05_CNN/            # Konvolučné siete
│   ├── 06_LSTM/           # Rekurentné siete
│   └── 07_Reinforcement_Learning/
│                          # Multi-armed bandit
│
└── parkinson_da/          # Semestrálny projekt
                           # → pozri parkinson_da/README.md
```

## Semestrálny projekt

**Detekcia Parkinsonovej choroby z reči — Domain Adaptation**

Porovnanie techník domain adaptation (DANN, MMD) pre klasifikáciu PD vs. Healthy
na akustických príznakoch z dvoch nezávislých datasetov (Oxford, Istanbul).

Detaily, inštalácia a spustenie → [`parkinson_da/README.md`](parkinson_da/README.md)

## Technológie

- Python 3.12
- PyTorch
- scikit-learn
- matplotlib

## Autor

**Mykyta Olym** — TUKE, 2025/2026
