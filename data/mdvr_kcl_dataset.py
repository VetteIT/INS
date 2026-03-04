"""
Dataset loader pre MDVR-KCL databazu.
MDVR-KCL (Mobile Device Voice Recordings at King's College London) je anglicky
dataset s recou pacientov s Parkinsonovou chorobou, nahravany na smartfone.

Zdroj: https://zenodo.org/record/2867216
Licencia: CC-BY 4.0

Struktura datasetu (po rozbaleni 26_29_09_2017_KCL.zip):
    MDVR-KCL/
        26-29_09_2017_KCL/          (alebo priamo ReadText/ a SpontaneousDialogue/)
            ReadText/
                HC/
                    ID00_hc_0_0_0.wav
                    ID01_hc_0_0_0.wav
                    ...
                PD/
                    ID02_pd_2_0_0.wav
                    ID04_pd_2_0_1.wav
                    ...
            SpontaneousDialogue/
                HC/
                    ID00_hc_0_0_0.wav
                    ...
                PD/
                    ID02_pd_2_0_0.wav
                    ...

Schema pomenovania suborov:
    ID{NN}_{hc|pd}_{H&Y}_{UPDRS_II-5}_{UPDRS_III-18}.wav
    - NN: cislo subjektu (00-36)
    - hc/pd: zdravy kontrol / Parkinson
    - H&Y: Hoehn & Yahr skore
    - UPDRS II-5: UPDRS II cast 5 skore
    - UPDRS III-18: UPDRS III cast 18 skore

Obsahuje 2 typy recovych uloh:
    - ReadText: citanie textu "The North Wind and the Sun"
    - SpontaneousDialogue: spontanny dialog

Ucastnici: 21 HC + 15 PD = 36 celkom

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import re
import glob

from data.base_dataset import BaseAudioDataset
from config.settings import MDVR_KCL_DIR


class MDVRKCLDataset(BaseAudioDataset):
    """
    Dataset trieda pre MDVR-KCL databazu.
    Obsahuje nahravky od PD pacientov a zdravych kontrol z King's College London.
    Nahravane na smartfone (Motorola Moto G4) pri 44.1 kHz.
    """

    def __init__(self, transform=None, feature_type="spectrogram", task_filter=None):
        """
        Parametre:
            transform: transformacie na audio
            feature_type: typ features ("spectrogram", "mfcc", "raw")
            task_filter: ak chceme len urcitu ulohu ("ReadText" alebo "SpontaneousDialogue")
        """
        self.task_filter = task_filter

        super().__init__(
            data_dir=MDVR_KCL_DIR,
            domain_name="MDVR-KCL",
            transform=transform,
            feature_type=feature_type
        )

    def _load_metadata(self):
        """
        Nacita metadata z MDVR-KCL datasetu.
        Dataset ma strukturu: {root}/[26-29_09_2017_KCL/]{ReadText,SpontaneousDialogue}/{HC,PD}/*.wav
        """
        if not os.path.exists(self.data_dir):
            print(f"  VAROVANIE: Priecinok {self.data_dir} neexistuje!")
            print(f"  Prosim stiahnite MDVR-KCL dataset z: https://zenodo.org/record/2867216")
            print(f"  Rozbalte ZIP a umiestnite obsah do: {self.data_dir}")
            return

        # Hladame korenovy priecinok - moze byt priamo data_dir alebo data_dir/26-29_09_2017_KCL
        root_dir = self._find_root_dir()
        if root_dir is None:
            print(f"  VAROVANIE: Nepodarilo sa najst strukturu MDVR-KCL v {self.data_dir}")
            print(f"  Ocakavana struktura: ReadText/{{HC,PD}}/*.wav a SpontaneousDialogue/{{HC,PD}}/*.wav")
            return

        # Definovanie uloh
        tasks = ["ReadText", "SpontaneousDialogue"]
        if self.task_filter:
            tasks = [t for t in tasks if t == self.task_filter]

        for task in tasks:
            task_dir = os.path.join(root_dir, task)
            if not os.path.exists(task_dir):
                print(f"  VAROVANIE: Priecinok {task_dir} neexistuje, preskakujem ulohu {task}")
                continue

            # Nacitame HC (Healthy Control) subory
            hc_dir = os.path.join(task_dir, "HC")
            if os.path.exists(hc_dir):
                self._load_from_dir(hc_dir, label=0, task_name=task)

            # Nacitame PD (Parkinson's Disease) subory
            pd_dir = os.path.join(task_dir, "PD")
            if os.path.exists(pd_dir):
                self._load_from_dir(pd_dir, label=1, task_name=task)

        if len(self.audio_paths) == 0:
            print(f"  VAROVANIE: Nenasli sa ziadne audio subory v {self.data_dir}")

    def _find_root_dir(self):
        """
        Najde korenovy priecinok s ReadText/ a SpontaneousDialogue/.
        Moze byt priamo v data_dir alebo v podpriecinku (napr. 26-29_09_2017_KCL/).
        Podporuje az dvojite vnorenie (napr. 26_29_09_2017_KCL/26-29_09_2017_KCL/).
        """
        # Skusime priamo data_dir
        if os.path.exists(os.path.join(self.data_dir, "ReadText")):
            return self.data_dir

        # Skusime podpriecinky (1. uroven)
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if os.path.isdir(subdir_path):
                if os.path.exists(os.path.join(subdir_path, "ReadText")):
                    return subdir_path

                # Skusime aj 2. uroven (dvojite vnorenie)
                for subsubdir in os.listdir(subdir_path):
                    subsubdir_path = os.path.join(subdir_path, subsubdir)
                    if os.path.isdir(subsubdir_path):
                        if os.path.exists(os.path.join(subsubdir_path, "ReadText")):
                            return subsubdir_path

        return None

    def _load_from_dir(self, directory, label, task_name):
        """
        Nacita vsetky WAV subory z daneho priecinka.

        Parametre:
            directory: cesta k priecinku s WAV subormi
            label: 0 = healthy, 1 = PD
            task_name: nazov ulohy (ReadText/SpontaneousDialogue)
        """
        for ext in ["*.wav", "*.WAV"]:
            files = glob.glob(os.path.join(directory, ext))
            for audio_file in sorted(files):
                self.audio_paths.append(audio_file)
                self.labels.append(label)

                # Extrahujeme speaker ID z nazvu suboru
                # Format: ID{NN}_{hc|pd}_{H&Y}_{UPDRS_II-5}_{UPDRS_III-18}.wav
                filename = os.path.basename(audio_file)
                speaker_id = self._extract_speaker_id(filename, task_name)
                self.speaker_ids.append(speaker_id)

    def _extract_speaker_id(self, filename, task_name):
        """
        Extrahuje speaker ID z nazvu suboru.

        Priklad: ID02_pd_2_0_0.wav -> KCL_ID02
                 ID00_hc_0_0_0.wav -> KCL_ID00
        """
        # Regex na zachytenie ID casti (napr. ID02, ID00)
        match = re.match(r"(ID\d+)", filename)
        if match:
            return "KCL_" + match.group(1)
        else:
            # Fallback - pouzijeme cely nazov bez pripony
            return "KCL_" + filename.split(".")[0]
