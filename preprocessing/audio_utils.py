"""
Pomocne funkcie pre spracovanie audio signalov.
Tu su zakladne operacie ako nacitanie, normalizacia, orezanie atd.

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np
import librosa
import torch

from config.settings import SAMPLE_RATE, SEGMENT_LENGTH


def load_and_preprocess_audio(audio_path, target_sr=None):
    """
    Nacita audio subor a spravi zakladny preprocessing.
    - nacita audio
    - prevzorkuje na cielovu vzorkovaciu frekvenciu
    - normalizuje amplitudu
    - odstrani ticho na zaciatku a konci
    
    Parametre:
        audio_path (str): cesta k audio suboru
        target_sr (int): cielova vzorkovacia frekvencia (default: SAMPLE_RATE z configu)
    
    Vrati:
        numpy array: preprocessovany audio signal
    """
    if target_sr is None:
        target_sr = SAMPLE_RATE
    
    # nacitame audio s librosa
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    
    # odstranime ticho na zaciatku a konci (trimming)
    # top_db=20 znamena ze vsetko tichsie nez 20dB pod maximom sa odstrani
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # normalizujeme amplitudu
    audio_normalized = normalize_audio(audio_trimmed)
    
    return audio_normalized


def normalize_audio(audio):
    """
    Normalizuje audio signal na rozsah [-1, 1].
    Toto je dolezite aby modely dostali data v rovnakom rozsahu.
    
    Parametre:
        audio (numpy array): audio signal
    
    Vrati:
        numpy array: normalizovany signal
    """
    # Najdeme maximum absolutnej hodnoty
    max_val = np.max(np.abs(audio))
    
    # Ak je audio uplne ticho (same nuly), vratime ho tak
    if max_val == 0:
        return audio
    
    # Normalizujeme
    normalized = audio / max_val
    
    return normalized


def pad_or_trim_audio(audio, target_length=None):
    """
    Upravi dlzku audio signalu na presnu dlzku.
    Ak je kratke - paddujeme nulami.
    Ak je dlhe - orezveme.
    
    Parametre:
        audio (numpy array): audio signal
        target_length (int): cielova dlzka v pocte vzoriek
    
    Vrati:
        numpy array: audio presne danej dlzky
    """
    if target_length is None:
        target_length = int(SEGMENT_LENGTH * SAMPLE_RATE)
    
    # Ak je presne spravna dlzka, nic nerobime
    if len(audio) == target_length:
        return audio
    
    # Ak je dlhe, orezveme
    if len(audio) > target_length:
        return audio[:target_length]
    
    # Ak je kratke, paddujeme nulami
    padded = np.zeros(target_length)
    padded[:len(audio)] = audio
    return padded


def segment_audio(audio, segment_length=None, overlap=0.5):
    """
    Rozreze audio na segmenty s prekrytim.
    
    Parametre:
        audio (numpy array): audio signal
        segment_length (float): dlzka segmentu v sekundach
        overlap (float): prekrytie medzi segmentami (0-1)
    
    Vrati:
        list: list numpy arrays (segmenty)
    """
    if segment_length is None:
        segment_length = SEGMENT_LENGTH
    
    # Pocet vzoriek na segment
    segment_samples = int(segment_length * SAMPLE_RATE)
    
    # Posun medzi segmentami
    hop = int(segment_samples * (1 - overlap))
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop
    
    # Ak sme neziskali ziadny segment (audio je kratke), paddujeme
    if len(segments) == 0:
        padded = pad_or_trim_audio(audio, segment_samples)
        segments.append(padded)
    
    return segments


def audio_to_tensor(audio):
    """
    Konvertuje numpy audio array na PyTorch tensor.
    Pridame aj channel dimenziu ak treba.
    
    Parametre:
        audio (numpy array): audio signal
    
    Vrati:
        torch.Tensor: tensor s audio datami
    """
    tensor = torch.FloatTensor(audio)
    
    # Ak je 1D, pridame channel dimenziu [1, length]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def add_noise(audio, noise_level=0.005):
    """
    Prida nahodny sum do audio signalu (data augmentation).
    Pouzivame biely sum s malou amplitudou.
    
    Parametre:
        audio (numpy array): povodny signal
        noise_level (float): uroven sumu (0.005 = 0.5%)
    
    Vrati:
        numpy array: signal so sumom
    """
    noise = np.random.randn(len(audio)) * noise_level
    noisy_audio = audio + noise
    return noisy_audio


def change_speed(audio, speed_factor=1.0):
    """
    Zmeni rychlost audio signalu (data augmentation).
    
    Parametre:
        audio (numpy array): povodny signal
        speed_factor (float): faktor rychlosti (1.0 = normalna, >1 = rychlejsie)
    
    Vrati:
        numpy array: signal so zmenenou rychlostou
    """
    # Pouzijeme librosa time_stretch
    stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
    return stretched
