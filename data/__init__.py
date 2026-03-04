# Data balicek - tu su vsetky moduly na nacitanie a spracovanie dat
# Kazdy dataset ma svoj vlastny modul

from data.base_dataset import BaseAudioDataset
from data.pcgita_dataset import PCGITADataset
from data.neurovoz_dataset import NeurovozDataset
from data.pdita_dataset import PDITADataset
from data.data_loader import create_data_loaders, get_domain_datasets
