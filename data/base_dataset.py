"""
Zakladna trieda pre vsetky datasety.
Vsetky datasety (PC-GITA, Neurovoz, PDITA) dedia z tejto triedy.
To nam usetri kopu kodu pretoze kazdy dataset ma podobne funkcie.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

# importujeme nase nastavenia
from config.settings import SAMPLE_RATE, SEGMENT_LENGTH, SEGMENT_OVERLAP


class BaseAudioDataset(Dataset):
    """
    Zakladna trieda pre audio datasety.
    Kazdy dataset (PC-GITA, Neurovoz, PDITA) bude diedit z tejto triedy.
    
    Tato trieda vie:
    - nacitat audio subory
    - rozrezat ich na segmenty
    - ulozit labely (PD alebo Healthy)
    - vrati data vo formate ktory potrebuje PyTorch
    """

    def __init__(self, data_dir, domain_name, transform=None, feature_type="spectrogram"):
        """
        Inicializacia datasetu.
        
        Parametre:
            data_dir (str): cesta k priecinku s datami
            domain_name (str): nazov domeny (napr. "PC-GITA")
            transform: transformacie ktore chceme aplikovat na data
            feature_type (str): typ prizvukov - "spectrogram", "mfcc", alebo "raw"
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.domain_name = domain_name
        self.transform = transform
        self.feature_type = feature_type
        
        # Tu si ulozime cesty k audio suborom a ich labely
        self.audio_paths = []    # cesty k wav suborom
        self.labels = []         # 0 = Healthy, 1 = PD (Parkinson)
        self.speaker_ids = []    # ID rečníka (kvoli speaker-independent split)
        
        # Toto musi implementovat kazdy konkretny dataset
        self._load_metadata()
        
        # Vypis info o datasete (tak vieme ci sa nacital spravne)
        print(f"[{self.domain_name}] Nacitanych {len(self.audio_paths)} nahrávok")
        print(f"  - Healthy: {self.labels.count(0)}")
        print(f"  - PD: {self.labels.count(1)}")

    def _load_metadata(self):
        """
        Nacita metadata o datasete - cesty k suborom a labely.
        Tuto metodu musi implementovat kazdy konkretny dataset,
        pretoze kazdy ma inu strukturu priecinkov.
        """
        raise NotImplementedError("Kazdy dataset musi implementovat _load_metadata!")

    def _load_audio(self, audio_path):
        """
        Nacita audio subor a prevzorkuje ho na SAMPLE_RATE.
        Pouzivame librosa, pretoze vie nacitat aj mp3, wav, flac...
        
        Parametre:
            audio_path (str): cesta k audio suboru
            
        Vrati:
            numpy array s audio signalom
        """
        try:
            # nacitame audio a prevzorkujeme na nas SAMPLE_RATE
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # Skontrolujeme ci audio nie je prazdne
            if len(audio) == 0:
                print(f"  VAROVANIE: Prazdny audio subor: {audio_path}")
                return None
            
            return audio
            
        except Exception as e:
            print(f"  CHYBA pri nacitani {audio_path}: {e}")
            return None

    def _segment_audio(self, audio):
        """
        Rozreze audio na segmenty pevnej dlzky.
        Toto robime aby sme mali vstupy rovnakej dlzky pre model.
        Ak je audio kratke, paddujeme ho nulami.
        
        Parametre:
            audio (numpy array): cely audio signal
            
        Vrati:
            list segmentov (kazdy je numpy array)
        """
        # Kolko vzoriek ma jeden segment
        segment_samples = int(SEGMENT_LENGTH * SAMPLE_RATE)
        
        # Ak je audio kratsie nez segment, paddujeme nulami
        if len(audio) < segment_samples:
            padded = np.zeros(segment_samples)
            padded[:len(audio)] = audio
            return [padded]
        
        # Rozrezeme audio na segmenty
        segments = []
        start = 0
        while start + segment_samples <= len(audio):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
            # posunieme sa o (1 - prekrytie) * dlzka segmentu
            start += int(segment_samples * (1 - SEGMENT_OVERLAP))
        
        return segments

    def __len__(self):
        """Vrati pocet vzoriek v datasete."""
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        Vrati jednu vzorku z datasetu.
        Toto vola PyTorch DataLoader ked treba data.
        
        Parametre:
            idx (int): index vzorky
            
        Vrati:
            tuple: (audio_data, label, domain_label)
        """
        # nacitame audio
        audio_path = self.audio_paths[idx]
        audio = self._load_audio(audio_path)
        
        # Ak sa audio nepodarilo nacitat, vratime nuly
        if audio is None:
            audio = np.zeros(int(SEGMENT_LENGTH * SAMPLE_RATE))
        
        # Orezeme alebo paddujeme na presnu dlzku
        target_length = int(SEGMENT_LENGTH * SAMPLE_RATE)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            padded = np.zeros(target_length)
            padded[:len(audio)] = audio
            audio = padded
        
        # Label pre klasyfikaciu (0=Healthy, 1=PD)
        label = torch.LongTensor([self.labels[idx]])[0]

        # Podla feature_type konvertujeme audio na spravny format
        if self.feature_type == "spectrogram":
            # Pre CNN - konvertujeme audio na mel spektrogram
            from preprocessing.spectrogram_maker import audio_to_spectrogram_tensor
            audio_tensor = audio_to_spectrogram_tensor(audio)  # [1, H, W]
        elif self.feature_type == "mfcc" or self.feature_type == "features":
            # Pre tradicne modely - extrahujeme akusticke prizvuky
            from preprocessing.feature_extraction import extract_acoustic_features
            features = extract_acoustic_features(audio)
            audio_tensor = torch.FloatTensor(features)
        else:
            # Raw audio - len konvertujeme na tensor
            audio_tensor = torch.FloatTensor(audio)

        # Ak mame nejake transformacie, aplikujeme ich
        if self.transform is not None:
            audio_tensor = self.transform(audio_tensor)

        return audio_tensor, label

    def get_domain_name(self):
        """Vrati nazov domeny."""
        return self.domain_name

    def get_labels(self):
        """Vrati vsetky labely."""
        return self.labels

    def get_speaker_ids(self):
        """Vrati vsetky speaker ID."""
        return self.speaker_ids
