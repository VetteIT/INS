"""
Dataset loader pre PDITA databazu.
PDITA je taliansky dataset s recou pacientov s Parkinsonovou chorobou.

Struktura datasetu (ocakavana):
    PDITA/
        healthy/
            H001.wav
            ...
        parkinsons/
            PD001.wav
            ...

Poznamka: PDITA moze mat inu strukturu priecinkov,
          ale princip nacitania je rovnaky.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import glob

from data.base_dataset import BaseAudioDataset
from config.settings import PDITA_DIR


class PDITADataset(BaseAudioDataset):
    """
    Dataset trieda pre PDITA databazu.
    Taliansky dataset s recovymi nahravkami PD pacientov.
    """

    def __init__(self, transform=None, feature_type="spectrogram"):
        """
        Parametre:
            transform: transformacie na audio
            feature_type: typ features
        """
        super().__init__(
            data_dir=PDITA_DIR,
            domain_name="PDITA",
            transform=transform,
            feature_type=feature_type
        )

    def _load_metadata(self):
        """
        Nacita metadata z PDITA datasetu.
        Skusime viacero moznych struktur priecinkov.
        """
        if not os.path.exists(self.data_dir):
            print(f"  VAROVANIE: Priecinok {self.data_dir} neexistuje!")
            print(f"  Prosim stiahnite PDITA dataset a umiestnite ho do: {self.data_dir}")
            return

        # Skusime najprv healthy/parkinsons strukturu
        healthy_dir = os.path.join(self.data_dir, "healthy")
        pd_dir = os.path.join(self.data_dir, "parkinsons")

        if os.path.exists(healthy_dir) and os.path.exists(pd_dir):
            self._load_standard_structure(healthy_dir, pd_dir)
        else:
            # Skusime strukturu s CSV metadatami
            self._load_with_csv_metadata()

    def _load_standard_structure(self, healthy_dir, pd_dir):
        """
        Nacita data zo standardnej struktury healthy/parkinsons.
        """
        # Healthy nahrávky
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            for audio_file in sorted(glob.glob(os.path.join(healthy_dir, ext))):
                self.audio_paths.append(audio_file)
                self.labels.append(0)  # healthy
                
                filename = os.path.basename(audio_file)
                speaker_id = "PDITA_H_" + filename.split(".")[0]
                self.speaker_ids.append(speaker_id)

        # PD nahrávky
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            for audio_file in sorted(glob.glob(os.path.join(pd_dir, ext))):
                self.audio_paths.append(audio_file)
                self.labels.append(1)  # PD
                
                filename = os.path.basename(audio_file)
                speaker_id = "PDITA_PD_" + filename.split(".")[0]
                self.speaker_ids.append(speaker_id)

    def _load_with_csv_metadata(self):
        """
        Ak dataset ma CSV subor s metadatami, pouzijeme ho.
        Hlada subor metadata.csv alebo labels.csv v hlavnom priecinku.
        """
        import csv
        
        # Skusime najst metadata CSV
        csv_candidates = [
            os.path.join(self.data_dir, "metadata.csv"),
            os.path.join(self.data_dir, "labels.csv"),
            os.path.join(self.data_dir, "participants.csv"),
        ]
        
        csv_path = None
        for candidate in csv_candidates:
            if os.path.exists(candidate):
                csv_path = candidate
                break
        
        if csv_path is not None:
            print(f"  Nasiel sa CSV subor: {csv_path}")
            
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # skusime rozne stlpce pre nazov suboru a label
                    filename = row.get("filename", row.get("file", row.get("audio_file", "")))
                    label_str = row.get("label", row.get("diagnosis", row.get("class", "")))
                    
                    if not filename:
                        continue
                    
                    audio_path = os.path.join(self.data_dir, filename)
                    if not os.path.exists(audio_path):
                        # skusime s .wav priponou
                        audio_path = os.path.join(self.data_dir, filename + ".wav")
                    
                    if os.path.exists(audio_path):
                        self.audio_paths.append(audio_path)
                        
                        # Urcime label
                        label_str = label_str.upper().strip()
                        if label_str in ["PD", "PARKINSON", "1", "PARKINSONS"]:
                            self.labels.append(1)
                        else:
                            self.labels.append(0)
                        
                        speaker_id = row.get("speaker_id", row.get("subject_id", filename.split(".")[0]))
                        self.speaker_ids.append("PDITA_" + str(speaker_id))
        else:
            # Posledna moznost - prehladame vsetky audio subory
            print(f"  Nenasiel sa CSV, skusim nacitat vsetky audio subory...")
            self._load_all_audio_files()

    def _load_all_audio_files(self):
        """
        Nacita vsetky audio subory a skusi urcit label z nazvu.
        Toto je posledna moznost ak nemame inu informaciu.
        """
        all_files = []
        for ext in ["*.wav", "*.WAV", "*.mp3", "*.flac"]:
            all_files.extend(glob.glob(os.path.join(self.data_dir, "**", ext), recursive=True))
        
        for audio_file in sorted(all_files):
            filename = os.path.basename(audio_file).upper()
            
            if "PD" in filename or "PARKINSON" in filename:
                self.audio_paths.append(audio_file)
                self.labels.append(1)
                self.speaker_ids.append("PDITA_" + os.path.basename(audio_file).split(".")[0])
            elif "H" in filename or "HC" in filename or "HEALTHY" in filename or "CTRL" in filename:
                self.audio_paths.append(audio_file)
                self.labels.append(0)
                self.speaker_ids.append("PDITA_" + os.path.basename(audio_file).split(".")[0])

        if len(self.audio_paths) == 0:
            print(f"  VAROVANIE: Nepodarilo sa nacitat ziadne data z {self.data_dir}")
