"""
Dataset loader pre Neurovoz databazu.
Neurovoz je spanielsky dataset s recou pacientov s Parkinsonovou chorobou.
Obsahuje rozne recove ulohy - sustained vowels, DDK, citanie a spontanna rec.

Struktura datasetu (ocakavana):
    Neurovoz/
        healthy/
            H_001.wav
            H_002.wav
            ...
        parkinsons/
            PD_001.wav
            PD_002.wav
            ...

Poznamka: Neurovoz ma trosku inu strukturu nez PC-GITA,
          preto ma vlastny loader.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import glob

from data.base_dataset import BaseAudioDataset
from config.settings import NEUROVOZ_DIR


class NeurovozDataset(BaseAudioDataset):
    """
    Dataset trieda pre Neurovoz databazu.
    Obsahuje nahrávky od PD pacientov a zdravych kontrol z Madridu.
    """

    def __init__(self, transform=None, feature_type="spectrogram"):
        """
        Parametre:
            transform: transformacie na audio
            feature_type: typ features
        """
        super().__init__(
            data_dir=NEUROVOZ_DIR,
            domain_name="Neurovoz",
            transform=transform,
            feature_type=feature_type
        )

    def _load_metadata(self):
        """
        Nacita zoznam audio suborov z Neurovoz datasetu.
        Neurovoz ma trosku inu strukturu - moze mat subpriecinky podla typu ulohy.
        """
        if not os.path.exists(self.data_dir):
            print(f"  VAROVANIE: Priecinok {self.data_dir} neexistuje!")
            print(f"  Prosim stiahnite Neurovoz dataset a umiestnite ho do: {self.data_dir}")
            return

        # Skusime najprv klasicku strukturu healthy/parkinsons
        healthy_dir = os.path.join(self.data_dir, "healthy")
        pd_dir = os.path.join(self.data_dir, "parkinsons")

        if os.path.exists(healthy_dir) and os.path.exists(pd_dir):
            # Klasicka struktura
            self._load_from_two_folders(healthy_dir, pd_dir)
        else:
            # Skusime alternativnu strukturu kde su subory pomiesane
            # a label je v nazve suboru (napr. H_001.wav vs PD_001.wav)
            self._load_from_filenames()

    def _load_from_two_folders(self, healthy_dir, pd_dir):
        """
        Nacita data z dvoch priecinkov - healthy a parkinsons.
        """
        # Healthy vzorky
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            healthy_files = glob.glob(os.path.join(healthy_dir, ext))
            for audio_file in sorted(healthy_files):
                self.audio_paths.append(audio_file)
                self.labels.append(0)  # healthy
                
                filename = os.path.basename(audio_file)
                # speaker ID je cislo v nazve
                speaker_id = "NV_H_" + filename.split(".")[0].split("_")[-1]
                self.speaker_ids.append(speaker_id)

        # PD vzorky
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            pd_files = glob.glob(os.path.join(pd_dir, ext))
            for audio_file in sorted(pd_files):
                self.audio_paths.append(audio_file)
                self.labels.append(1)  # PD
                
                filename = os.path.basename(audio_file)
                speaker_id = "NV_PD_" + filename.split(".")[0].split("_")[-1]
                self.speaker_ids.append(speaker_id)

    def _load_from_filenames(self):
        """
        Ak data su v jednom priecinku, rozlisime healthy/PD podla nazvu suboru.
        H_ alebo HC_ = healthy, PD_ = parkinsons
        """
        all_files = []
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            all_files.extend(glob.glob(os.path.join(self.data_dir, "**", ext), recursive=True))
        
        for audio_file in sorted(all_files):
            filename = os.path.basename(audio_file).upper()
            
            # Urcime label podla nazvu suboru
            if filename.startswith("H") or "HEALTHY" in filename or "HC" in filename:
                self.audio_paths.append(audio_file)
                self.labels.append(0)  # healthy
                speaker_id = "NV_" + os.path.basename(audio_file).split(".")[0]
                self.speaker_ids.append(speaker_id)
                
            elif filename.startswith("PD") or "PARKINSON" in filename:
                self.audio_paths.append(audio_file)
                self.labels.append(1)  # PD
                speaker_id = "NV_" + os.path.basename(audio_file).split(".")[0]
                self.speaker_ids.append(speaker_id)
            
            # Ak nevieme urcit label, preskocime subor
            else:
                print(f"  Preskakujem {audio_file} - neviem urcit label")

        if len(self.audio_paths) == 0:
            print(f"  VAROVANIE: Nenasli sa ziadne spravne pomenovane subory v {self.data_dir}")
