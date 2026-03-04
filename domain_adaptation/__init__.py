# Domain adaptation balicek
# Tu su vsetky implementacie domain adaptation technik
# Kazda technika ma svoj vlastny subor

from domain_adaptation.baseline import BaselineTrainer
from domain_adaptation.dann import DANNTrainer
from domain_adaptation.mmd_adaptation import MMDTrainer
from domain_adaptation.contrastive_alignment import ContrastiveTrainer
from domain_adaptation.multi_source_adaptation import MultiSourceTrainer
