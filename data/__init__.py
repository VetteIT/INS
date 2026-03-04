# Data balicek - tu su vsetky moduly na nacitanie a spracovanie dat
# Kazdy dataset ma svoj vlastny modul

from data.base_dataset import BaseAudioDataset
from data.mdvr_kcl_dataset import MDVRKCLDataset
from data.italian_pvs_dataset import ItalianPVSDataset
from data.data_loader import create_data_loaders, get_domain_datasets
