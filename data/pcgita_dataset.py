"""
Dataset loader pre PC-GITA databazu.
PC-GITA je kolumbijsky dataset s recou pacientov s Parkinsonovou chorobou.
Obsahuje rozne recove ulohy - sustained vowels, DDK, read text, monologue.

Struktura datasetu (ocakavana):
    PC-GITA/
        healthy/
            HC001_vowel_a.wav
            HC001_read_text.wav
            ...
        parkinsons/
            PD001_vowel_a.wav
            PD001_read_text.wav
            ...

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import glob

from data.base_dataset import BaseAudioDataset
from config.settings import PCGITA_DIR


class PCGITADataset(BaseAudioDataset):
    """
    Dataset trieda pre PC-GITA databazu.
    Obsahuje recove nahrávky od 50 PD pacientov a 50 zdravych kontrol.
    Recove ulohy: sustained vowels (/a/, /i/, /u/), DDK, citanie textu, monolog.
    """

    def __init__(self, transform=None, feature_type="spectrogram", task_filter=None):
        """
        Parametre:
            transform: transformacie na audio
            feature_type: typ features ("spectrogram", "mfcc", "raw")
            task_filter: ak chceme len urcitu ulohu (napr. "vowel_a")
        """
        self.task_filter = task_filter
        
        super().__init__(
            data_dir=PCGITA_DIR,
            domain_name="PC-GITA",
            transform=transform,
            feature_type=feature_type
        )

    def _load_metadata(self):
        """
        Nacita zoznam audio suborov a ich labely z PC-GITA datasetu.
        Prechadzame priecinky 'healthy' a 'parkinsons'.
        """
        # Najprv skontrolujeme ci existuje priecinok s datami
        if not os.path.exists(self.data_dir):
            print(f"  VAROVANIE: Priecinok {self.data_dir} neexistuje!")
            print(f"  Prosim stiahnite PC-GITA dataset a umiestnite ho do: {self.data_dir}")
            # Nevyhadzujeme chybu, proste dataset bude prazdny
            return

        # Nacitame healthy vzorky
        healthy_dir = os.path.join(self.data_dir, "healthy")
        if os.path.exists(healthy_dir):
            healthy_files = glob.glob(os.path.join(healthy_dir, "*.wav"))
            
            # Ak chceme len urcitu ulohu, filtrujeme
            if self.task_filter is not None:
                healthy_files = [f for f in healthy_files if self.task_filter in os.path.basename(f)]
            
            for audio_file in sorted(healthy_files):
                self.audio_paths.append(audio_file)
                self.labels.append(0)  # 0 = healthy
                
                # Skusime extrahovat speaker ID z nazvu suboru
                filename = os.path.basename(audio_file)
                speaker_id = filename.split("_")[0]  # napr. HC001
                self.speaker_ids.append(speaker_id)

        # Nacitame parkinsons vzorky
        pd_dir = os.path.join(self.data_dir, "parkinsons")
        if os.path.exists(pd_dir):
            pd_files = glob.glob(os.path.join(pd_dir, "*.wav"))
            
            if self.task_filter is not None:
                pd_files = [f for f in pd_files if self.task_filter in os.path.basename(f)]
            
            for audio_file in sorted(pd_files):
                self.audio_paths.append(audio_file)
                self.labels.append(1)  # 1 = PD
                
                filename = os.path.basename(audio_file)
                speaker_id = filename.split("_")[0]
                self.speaker_ids.append(speaker_id)

        # Kontrola ci sme nasli nejake data
        if len(self.audio_paths) == 0:
            print(f"  VAROVANIE: Nenasli sa ziadne audio subory v {self.data_dir}")
            print(f"  Skontrolujte strukturu priecinkov (healthy/ a parkinsons/)")
