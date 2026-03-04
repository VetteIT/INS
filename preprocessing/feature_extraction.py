"""
Extrakcia akustickych prizvukov (features) z audio signalu.
Tu extrahujeme rozne charakteristiky hlasu ktore sa pouzivaju
na detekciu Parkinsonovej choroby:

- MFCC (Mel-Frequency Cepstral Coefficients) - najdôleżitejsie prizvuky
- Jitter - variabilita zakladnej frekvencie (F0)
- Shimmer - variabilita amplitudy
- HNR (Harmonics-to-Noise Ratio) - pomer harmonickych k sumu
- a dalsie...

Prizvuky su cislene hodnoty ktore popisuju vlastnosti hlasu.
PD pacienti maju typicky vyssiu variabilitu (jitter, shimmer) a nizsie HNR.

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np
import librosa

from config.settings import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, F0_MIN, F0_MAX


def extract_mfcc(audio, sr=None, n_mfcc=None):
    """
    Extrahuje MFCC (Mel-Frequency Cepstral Coefficients) z audio signalu.
    MFCC su najrozsirenejsie prizvuky v spracovani reci.
    Zachytavaji spektralny obal hlasu.
    
    Parametre:
        audio (numpy array): audio signal
        sr (int): vzorkovacia frekvencia
        n_mfcc (int): pocet MFCC koeficientov
    
    Vrati:
        numpy array: MFCC prizvuky [n_mfcc, time_steps]
    """
    if sr is None:
        sr = SAMPLE_RATE
    if n_mfcc is None:
        n_mfcc = N_MFCC
    
    # Vypočítame MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    
    return mfcc


def extract_mfcc_stats(audio, sr=None, n_mfcc=None):
    """
    Extrahuje statistiky z MFCC - priemer, smerodajna odchylka, min, max.
    Toto pouzivame pre tradicne ML modely (MLP) ktore
    potrebuju vstup s pevnou velkostou.
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
        n_mfcc: pocet MFCC
    
    Vrati:
        numpy array: vektor prizvukov (n_mfcc * 4 hodnot)
    """
    mfcc = extract_mfcc(audio, sr, n_mfcc)
    
    # Pre kazdy MFCC koeficient spocitame statistiky
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    
    # Delta MFCC (zmena MFCC v case) - tiez dolezite
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
    delta_mfcc_std = np.std(delta_mfcc, axis=1)
    
    # Delta-delta MFCC (zrychlenie zmeny)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
    delta2_mfcc_std = np.std(delta2_mfcc, axis=1)
    
    # Spojime vsetko do jedneho vektora
    features = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_min, mfcc_max,
        delta_mfcc_mean, delta_mfcc_std,
        delta2_mfcc_mean, delta2_mfcc_std,
    ])
    
    return features


def extract_f0(audio, sr=None):
    """
    Extrahuje zakladnu frekvenciu (F0/pitch) z audio signalu.
    F0 je frekvencia kmitania hlasiviek - u PD pacientov
    je casto menej stabilna.
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
    
    Vrati:
        numpy array: F0 hodnoty v case
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # Pouzijeme librosa.pyin na estimaciu F0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sr,
    )
    
    # Odstranime NaN hodnoty (kde nie je rec)
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) == 0:
        # Ak sme nenasli ziadnu F0, vratime prazdne pole
        return np.array([0.0])
    
    return f0_clean


def compute_jitter(f0_values):
    """
    Vypocita jitter (variabilitu F0) z F0 hodnot.
    Jitter meria ako moc sa meni zakladna frekvencia medzi po sebe
    iducimi periodami. Vyssí jitter = menej stabilny hlas = mozna PD.
    
    Parametre:
        f0_values: pole F0 hodnot
    
    Vrati:
        float: jitter hodnota (v percentach)
    """
    if len(f0_values) < 2:
        return 0.0
    
    # Periody = 1/F0
    periods = 1.0 / f0_values
    
    # Jitter (lokalna) = priemer absolutnych rozdielov po sebe iducich periood
    # normalizovany priemernou periodou
    diffs = np.abs(np.diff(periods))
    jitter = np.mean(diffs) / np.mean(periods) * 100  # v percentach
    
    return jitter


def compute_shimmer(audio, sr=None):
    """
    Vypocita shimmer (variabilitu amplitudy) z audio signalu.
    Shimmer meria ako moc sa meni sila hlasu.
    Vyssi shimmer u PD pacientov.
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
    
    Vrati:
        float: shimmer hodnota (v percentach)
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # Rozdelime signal na kratke ramce
    frame_length = int(0.025 * sr)  # 25ms ramce
    hop = int(0.010 * sr)           # 10ms hop
    
    # Spocitame amplitudu (RMS) pre kazdy ramec
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop)
    amplitudes = np.sqrt(np.mean(frames ** 2, axis=0))
    
    if len(amplitudes) < 2:
        return 0.0
    
    # Shimmer = priemer absolutnych rozdielov amplitud / priemerna amplituda
    diffs = np.abs(np.diff(amplitudes))
    shimmer = np.mean(diffs) / np.mean(amplitudes) * 100  # v percentach
    
    return shimmer


def compute_hnr(audio, sr=None):
    """
    Vypocita HNR (Harmonics-to-Noise Ratio).
    HNR meria pomer harmonickych zloziek k sumu v hlase.
    Nizsie HNR = viac sumu v hlase = mozny PD.
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
    
    Vrati:
        float: HNR hodnota v dB
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # Jednoduchy odhad HNR pomocou autokorelacnej metody
    # Toto nie je uplne presne ale pre nas ucel staci
    
    # Autocorrelacia
    n = len(audio)
    if n < 2:
        return 0.0
    
    # Normalizovana autokorelácia
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[n-1:]  # vezmeme len kladne oneskorenia
    autocorr = autocorr / autocorr[0]  # normalizujeme
    
    # Najdeme maximum autokorelacnej funkcie (okrem lag=0)
    # v rozsahu ocakavanych periood hlasu
    min_lag = int(sr / F0_MAX)  # minimalne oneskorenie
    max_lag = int(sr / F0_MIN)  # maximalne oneskorenie
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
    
    if min_lag >= max_lag:
        return 0.0
    
    search_region = autocorr[min_lag:max_lag]
    
    if len(search_region) == 0:
        return 0.0
    
    max_autocorr = np.max(search_region)
    
    # HNR v dB
    if max_autocorr <= 0 or max_autocorr >= 1:
        return 0.0
    
    hnr = 10 * np.log10(max_autocorr / (1 - max_autocorr))
    
    return hnr


def extract_spectral_features(audio, sr=None):
    """
    Extrahuje spektralne prizvuky z audio signalu.
    Tieto prizvuky popisuju frekvenčne vlastnosti hlasu.
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
    
    Vrati:
        dict: slovnik s prizvukmi
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    features = {}
    
    # Spectral centroid - "stred masy" spektra
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features["spectral_centroid_mean"] = np.mean(spec_centroid)
    features["spectral_centroid_std"] = np.std(spec_centroid)
    
    # Spectral bandwidth - sirka spektra
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(spec_bw)
    features["spectral_bandwidth_std"] = np.std(spec_bw)
    
    # Spectral rolloff - frekvencia pod ktorou je 85% energie
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features["spectral_rolloff_mean"] = np.mean(spec_rolloff)
    features["spectral_rolloff_std"] = np.std(spec_rolloff)
    
    # Zero crossing rate - kolkokrat signal pretne nulovu os
    zcr = librosa.feature.zero_crossing_rate(audio)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    
    # RMS energia
    rms = librosa.feature.rms(y=audio)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    
    return features


def extract_acoustic_features(audio, sr=None):
    """
    Extrahuje VSETKY akusticke prizvuky z audio signalu.
    Toto je hlavna funkcia ktora spoji vsetky prizvuky do jedneho vektora.
    
    Pouzivame ju pre tradicne ML modely (MLP).
    
    Parametre:
        audio: audio signal
        sr: vzorkovacia frekvencia
    
    Vrati:
        numpy array: vektor vsetkych prizvukov
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    all_features = []
    
    # 1. MFCC statistiky (najdolezitejsie)
    mfcc_stats = extract_mfcc_stats(audio, sr)
    all_features.extend(mfcc_stats)
    
    # 2. F0 statistiky
    f0 = extract_f0(audio, sr)
    all_features.append(np.mean(f0))   # priemerna F0
    all_features.append(np.std(f0))    # variabilita F0
    all_features.append(np.min(f0))    # minimalna F0
    all_features.append(np.max(f0))    # maximalna F0
    
    # 3. Jitter
    jitter = compute_jitter(f0)
    all_features.append(jitter)
    
    # 4. Shimmer
    shimmer = compute_shimmer(audio, sr)
    all_features.append(shimmer)
    
    # 5. HNR
    hnr = compute_hnr(audio, sr)
    all_features.append(hnr)
    
    # 6. Spektralne prizvuky
    spectral = extract_spectral_features(audio, sr)
    all_features.extend(spectral.values())
    
    # Konvertujeme na numpy array
    feature_vector = np.array(all_features, dtype=np.float32)
    
    # Nahradime pripadne NaN a Inf hodnoty nulami
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector
