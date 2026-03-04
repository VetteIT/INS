"""
Modul na vytvaranie spektrogramov z audio signalov.
Spektrogramy pouzivame ako vstup pre CNN model.
Spektrogram je vlastne obrazok ktory ukazuje frekvencie v case.

Typy spektrogramov:
- Obycajny spektrogram (STFT)
- Mel spektrogram (pouziva mel skalu ktora je blizsie ludskemu vnimaniu)
- Log-mel spektrogram (logaritmicky mel spektrogram - najcastejsi)

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np
import librosa
import torch

from config.settings import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, SPEC_HEIGHT, SPEC_WIDTH


def create_spectrogram(audio, sr=None):
    """
    Vytvori obycajny spektrogram (STFT) z audio signalu.
    
    Parametre:
        audio (numpy array): audio signal
        sr (int): vzorkovacia frekvencia
    
    Vrati:
        numpy array: spektrogram [freq_bins, time_steps]
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # STFT = Short-Time Fourier Transform
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Absolutna hodnota (magnitude) a konvertujeme na dB
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    return spectrogram_db


def create_mel_spectrogram(audio, sr=None):
    """
    Vytvori mel spektrogram z audio signalu.
    Mel skala lepsie odpoveda ludskemu vnimaniu zvuku.
    Toto je najcastejsi typ spektrogramu v spracovani reci.
    
    Parametre:
        audio (numpy array): audio signal
        sr (int): vzorkovacia frekvencia
    
    Vrati:
        numpy array: mel spektrogram [n_mels, time_steps]
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    
    # Konvertujeme na decibelovu skalu (logaritmicka)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def spectrogram_to_image(spectrogram, height=None, width=None):
    """
    Konvertuje spektrogram na obrazok s pevnou velkostou.
    Toto potrebujeme pre CNN ktory ocakava vstup pevnej velkosti.
    
    Parametre:
        spectrogram (numpy array): spektrogram
        height (int): vyska vystupneho obrazku
        width (int): sirka vystupneho obrazku
    
    Vrati:
        numpy array: resizovany spektrogram [height, width]
    """
    if height is None:
        height = SPEC_HEIGHT
    if width is None:
        width = SPEC_WIDTH
    
    # Normalizujeme na rozsah [0, 1]
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    
    if spec_max - spec_min > 0:
        normalized = (spectrogram - spec_min) / (spec_max - spec_min)
    else:
        normalized = np.zeros_like(spectrogram)
    
    # Resizujeme na pozadovanu velkost
    # Pouzijeme jednoduche nearest neighbor interpolaciu
    from PIL import Image
    
    # Konvertujeme na PIL Image
    img = Image.fromarray((normalized * 255).astype(np.uint8))
    img_resized = img.resize((width, height), Image.BILINEAR)
    
    # Konvertujeme spat na numpy array a normalizujeme na [0, 1]
    result = np.array(img_resized, dtype=np.float32) / 255.0
    
    return result


def audio_to_spectrogram_tensor(audio, sr=None, spec_type="mel"):
    """
    Konvertuje audio signal priamo na PyTorch tensor spektrogramu.
    Toto je hlavna funkcia ktoru volame pri trenovani CNN.
    
    Parametre:
        audio (numpy array alebo torch.Tensor): audio signal
        sr (int): vzorkovacia frekvencia
        spec_type (str): typ spektrogramu - "mel" alebo "stft"
    
    Vrati:
        torch.Tensor: spektrogram tensor [1, height, width]
    """
    if sr is None:
        sr = SAMPLE_RATE
    
    # Ak je to tensor, konvertujeme na numpy
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    # Ak je 2D (batch), vezmeme prvy
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Vytvorime spektrogram podla typu
    if spec_type == "mel":
        spec = create_mel_spectrogram(audio, sr)
    else:
        spec = create_spectrogram(audio, sr)
    
    # Resizujeme na fixnu velkost
    spec_image = spectrogram_to_image(spec)
    
    # Konvertujeme na tensor a pridame channel dimenziu
    spec_tensor = torch.FloatTensor(spec_image).unsqueeze(0)  # [1, H, W]
    
    return spec_tensor
