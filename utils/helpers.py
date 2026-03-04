"""
Pomocne funkcie ktore sa pouzivaju na roznych miestach v projekte.
Tu su drobnosti ktore nemaju vlastny modul.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import torch
import numpy as np
import random


def get_device():
    """
    Zisti ci mame GPU a vrati spravne zariadenie.
    Ak mame NVIDIA GPU s CUDA, pouzijeme ju (rychlejsie trenovanie).
    Inak pouzijeme CPU.
    
    Vrati:
        torch.device: zariadenie na vypocty
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Pouzivam GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("GPU nie je dostupna, pouzivam CPU")
    
    return device


def set_all_seeds(seed=42):
    """
    Nastavi seed pre vsetky kniznice.
    Toto je dolezite pre reprodukovatelnost vysledkov.
    Ked spustime experiment viackrat s rovnakym seedom,
    dostaneme rovnake vysledky.
    
    Parametre:
        seed (int): hodnota seedu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Toto spomaluje ale zarucuje reprodukovatelnost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Vsetky seeds nastavene na: {seed}")


def ensure_dir(path):
    """
    Vytvori priecinok ak neexistuje.
    
    Parametre:
        path (str): cesta k priecinku
    """
    os.makedirs(path, exist_ok=True)


def count_parameters(model):
    """
    Spocita pocet parametrov v modeli.
    Uzitocne na porovnanie velkosti modelov.
    
    Parametre:
        model (nn.Module): PyTorch model
    
    Vrati:
        tuple: (celkovy pocet, trenovatelny pocet)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Celkovy pocet parametrov: {total:,}")
    print(f"  Trenovatelnych parametrov: {trainable:,}")
    
    return total, trainable


def format_time(seconds):
    """
    Formatuje cas z sekund na citatelny format.
    
    Parametre:
        seconds (float): cas v sekundach
    
    Vrati:
        str: formatovany cas (napr. "2m 35s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_experiment_header(experiment_name, model_type, da_method,
                            source_domain, target_domain):
    """
    Vypise peknu hlavicku pre experiment.
    Len kozmeticka vec ale pomaha pri citani logov.
    
    Parametre:
        experiment_name: nazov experimentu
        model_type: typ modelu
        da_method: DA technika
        source_domain: zdrojova domena
        target_domain: cielova domena
    """
    print("\n")
    print("=" * 70)
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"  Model:          {model_type}")
    print(f"  DA technika:    {da_method}")
    print(f"  Source domena:   {source_domain}")
    print(f"  Target domena:  {target_domain}")
    print("=" * 70)


def save_results_to_csv(results_dict, output_path):
    """
    Ulozi vysledky do CSV suboru.
    Kazdy riadok = jeden experiment.
    
    Parametre:
        results_dict (dict): {experiment_name: {metrics...}}
        output_path (str): cesta k vystupnemu CSV
    """
    import csv
    
    ensure_dir(os.path.dirname(output_path))
    
    # Zistime vsetky metriky
    all_metric_names = set()
    for exp_results in results_dict.values():
        for key in exp_results:
            if isinstance(exp_results[key], (int, float)):
                all_metric_names.add(key)
    
    metric_names = sorted(all_metric_names)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Hlavicka
        header = ["Experiment"] + metric_names
        writer.writerow(header)
        
        # Data
        for exp_name, metrics in results_dict.items():
            row = [exp_name]
            for metric in metric_names:
                value = metrics.get(metric, "N/A")
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            writer.writerow(row)
    
    print(f"  Vysledky ulozene do: {output_path}")
