"""
Tyzdne 1-2: Stiahnutie a priprava datasetov.
Vzor: Cvicenie 2 - Praca s datami.ipynb (stahovanie datasetov, transformacie)

Dva verejne datasety z UCI ML Repository (oba CC BY 4.0):

Domain A - Oxford Parkinson's Dataset (Little et al., 2008)
  https://archive.ics.uci.edu/dataset/174/parkinsons
  195 vzoriek, 31 pacientov (23 PD + 8 Healthy)

Domain B - Parkinson Replicated Acoustic Features (Naranjo et al., 2016)
  https://archive.ics.uci.edu/dataset/489
  240 vzoriek, 80 subjektov (40 PD + 40 Healthy), 3 nahravky/subjekt

Vyberame 12 spolocnych akustickych priznakov:
  4x Jitter (perturbacia zakladnej frekvencie)
  4x Shimmer (perturbacia amplitudy)
  1x HNR (pomer harmonickych ku sumu)
  3x nelinearne (RPDE, DFA, PPE)
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd

from config import DATA_DIR


# ---- Mapovanie stlpcov z roznych datasetov na spolocne nazvy ----
# Overene stlpce z realnych CSV suborov (stiahnutie a skontrolovane)

# Dataset 174: Oxford Parkinson's
# Priamy CSV: https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data
OXFORD_MAP = {
    'MDVP:Jitter(%)': 'jitter_rel',
    'MDVP:Jitter(Abs)': 'jitter_abs',
    'MDVP:RAP': 'jitter_rap',
    'MDVP:PPQ': 'jitter_ppq',
    'MDVP:Shimmer': 'shimmer',
    'MDVP:Shimmer(dB)': 'shimmer_db',
    'Shimmer:APQ3': 'shimmer_apq3',
    'Shimmer:APQ5': 'shimmer_apq5',
    'HNR': 'hnr',
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'PPE': 'ppe',
}

# Dataset 489: Istanbul Replicated Acoustic Features
# ZIP z: https://archive.ics.uci.edu/static/public/489/
# CSV vnutri: ReplicatedAcousticFeatures-ParkinsonDatabase.csv
ISTANBUL_MAP = {
    'Jitter_rel': 'jitter_rel',
    'Jitter_abs': 'jitter_abs',
    'Jitter_RAP': 'jitter_rap',
    'Jitter_PPQ': 'jitter_ppq',
    'Shim_loc': 'shimmer',
    'Shim_dB': 'shimmer_db',
    'Shim_APQ3': 'shimmer_apq3',
    'Shim_APQ5': 'shimmer_apq5',
    'HNR05': 'hnr',       # HNR v pasme 0-500 Hz (najblizsi k HNR z Oxford)
    'RPDE': 'rpde',
    'DFA': 'dfa',
    'PPE': 'ppe',
}


def download_oxford():
    """
    Stiahne Oxford Parkinson's Dataset (UCI #174).
    Priamy download CSV - najspolahlivejsia metoda.
    https://archive.ics.uci.edu/dataset/174/parkinsons
    """
    print("[Domain A] Oxford Parkinson's Dataset (Little et al., 2008)")

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/parkinsons/parkinsons.data")
    df = pd.read_csv(url)

    # Oddelenie labelov
    labels = df['status'].values

    # Vyberieme a premenujeme spolocne priznaky
    common = list(OXFORD_MAP.values())
    df_out = df.rename(columns=OXFORD_MAP)[common].copy()
    df_out['label'] = labels
    df_out['domain'] = 'oxford'

    print(f"  Pocet vzoriek: {len(df_out)}")
    print(f"  PD: {(df_out['label']==1).sum()}, "
          f"Healthy: {(df_out['label']==0).sum()}")
    return df_out


def download_istanbul():
    """
    Stiahne Istanbul Replicated Acoustic Features (UCI #489).
    Stiahne ZIP subor a extrahuje CSV.
    https://archive.ics.uci.edu/dataset/489
    """
    print("\n[Domain B] Istanbul Replicated Features (Naranjo et al., 2016)")

    zip_url = ("https://archive.ics.uci.edu/static/public/489/"
               "parkinson+dataset+with+replicated+acoustic+features.zip")

    # Stiahnutie a rozbalenie ZIP
    resp = urllib.request.urlopen(zip_url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))

    # Najdeme CSV subor v ZIPE
    csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
    with z.open(csv_name) as csv_file:
        df = pd.read_csv(csv_file)

    # Labely: Status stlpec (0=Healthy, 1=PD)
    labels = df['Status'].values

    # Vyberieme a premenujeme spolocne priznaky
    common = list(ISTANBUL_MAP.values())
    df_out = df.rename(columns=ISTANBUL_MAP)[common].copy()
    df_out['label'] = labels
    df_out['domain'] = 'istanbul'

    print(f"  Pocet vzoriek: {len(df_out)}")
    print(f"  PD: {(df_out['label']==1).sum()}, "
          f"Healthy: {(df_out['label']==0).sum()}")
    return df_out


def download_datasets():
    """Stiahne vsetky datasety a ulozi ich do CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("Stahovanie datasetov z UCI ML Repository")
    print("=" * 60)

    df_oxford = download_oxford()
    df_istanbul = download_istanbul()

    # Ulozenie do CSV
    df_oxford.to_csv(os.path.join(DATA_DIR, 'oxford.csv'), index=False)
    df_istanbul.to_csv(os.path.join(DATA_DIR, 'istanbul.csv'), index=False)

    print(f"\nData ulozene do: {DATA_DIR}/")
    print(f"  oxford.csv   ({len(df_oxford)} vzoriek)")
    print(f"  istanbul.csv ({len(df_istanbul)} vzoriek)")

    return df_oxford, df_istanbul


if __name__ == '__main__':
    download_datasets()
