"""
============================================================================
HLAVNY VSTUPNY BOD PROJEKTU
============================================================================

Porovnanie domain adaptation techník pre detekciu Parkinsonovej choroby z reči

Tento skript je vstupny bod celeho projektu. Spustite ho pomocou:
    python main.py

Argumenty:
    --model       Typ modelu: cnn, traditional, wav2vec (default: cnn)
    --method      DA technika: baseline, dann, mmd, contrastive, multi_source, all
    --source      Source domena: PC-GITA, Neurovoz, PDITA
    --target      Target domena: PC-GITA, Neurovoz, PDITA
    --epochs      Pocet epoch (default: 50)
    --run-all     Spusti vsetky experimenty

Priklady:
    python main.py --model cnn --method baseline --source PC-GITA --target Neurovoz
    python main.py --model cnn --method dann --source PC-GITA --target PDITA --epochs 30
    python main.py --run-all --epochs 20

Autori: Dmytro Protsun (dmytro.protsun@student.tuke.sk)
        Mykyta Olym (mykyta.olym@student.tuke.sk)

Predmet: INS
Semestrálny projekt 2025/2026
TUKE - Technická univerzita v Košiciach
============================================================================
"""

import argparse
import sys
import os
import time

# Pridame korenoby priecinok do Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DOMAIN_NAMES, ADAPTATION_METHODS, MODEL_TYPES
from experiments.run_all_experiments import ExperimentRunner
from experiments.compare_results import ResultsComparator
from utils.helpers import set_all_seeds, get_device, format_time


def parse_arguments():
    """
    Parsuje argumenty z prikazoveho riadku.
    Toto nam umoznuje spustat rozne experimenty bez zmeny kodu.
    """
    parser = argparse.ArgumentParser(
        description="Porovnanie domain adaptation technik pre detekciu PD z reci"
    )
    
    parser.add_argument(
        "--model", type=str, default="cnn",
        choices=["cnn", "traditional", "wav2vec"],
        help="Typ modelu: cnn, traditional (SVM/MLP), wav2vec (default: cnn)"
    )
    
    parser.add_argument(
        "--method", type=str, default="baseline",
        choices=["baseline", "dann", "mmd", "contrastive", "multi_source", "all"],
        help="Domain adaptation technika (default: baseline)"
    )
    
    parser.add_argument(
        "--source", type=str, default="PC-GITA",
        choices=DOMAIN_NAMES,
        help="Source (zdrojova) domena"
    )
    
    parser.add_argument(
        "--target", type=str, default="Neurovoz",
        choices=DOMAIN_NAMES,
        help="Target (cielova) domena"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Pocet trenoavacich epoch (default: 50)"
    )
    
    parser.add_argument(
        "--run-all", action="store_true",
        help="Spusti VSETKY experimenty (vsetky kombinacie)"
    )
    
    parser.add_argument(
        "--quick", action="store_true",
        help="Rychly mod - menej epoch, len baseline a DANN (na testovanie)"
    )
    
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed pre reprodukovatelnost (default: 42)"
    )
    
    return parser.parse_args()


def run_quick_test():
    """
    Rychly test na overenie ze vsetko funguje.
    Spusti len par experimentov s malym poctom epoch.
    Uzitocne na overenie ze kod sa niekde nesype.
    """
    print("\n" + "=" * 70)
    print("  RYCHLY TEST (few epochs, basic experiments)")
    print("=" * 70)
    
    runner = ExperimentRunner(
        model_types=["cnn"],
        da_methods=["baseline", "dann"],
        domains=DOMAIN_NAMES,
    )
    
    # Spustime len jeden experiment
    results = runner.run_single_experiment(
        model_type="cnn",
        da_method="baseline",
        source_domain="PC-GITA",
        target_domain="Neurovoz",
        num_epochs=5,
    )
    
    if results is not None:
        print("\n  Rychly test USPESNY!")
    else:
        print("\n  Rychly test prebehol (mozno chybaju data)")
    
    return results


def run_single(args):
    """
    Spusti jeden experiment podla argumentov.
    """
    print(f"\n  Spustam experiment:")
    print(f"    Model:  {args.model}")
    print(f"    Metoda: {args.method}")
    print(f"    Source: {args.source}")
    print(f"    Target: {args.target}")
    print(f"    Epochy: {args.epochs}")
    
    runner = ExperimentRunner(
        model_types=[args.model],
        da_methods=[args.method],
    )
    
    if args.method == "multi_source":
        source_domains = [d for d in DOMAIN_NAMES if d != args.target]
        results = runner.run_multi_source_experiment(
            model_type=args.model,
            source_domains=source_domains,
            target_domain=args.target,
            num_epochs=args.epochs,
        )
    else:
        results = runner.run_single_experiment(
            model_type=args.model,
            da_method=args.method,
            source_domain=args.source,
            target_domain=args.target,
            num_epochs=args.epochs,
        )
    
    return results


def run_all(args):
    """
    Spusti vsetky experimenty.
    """
    runner = ExperimentRunner()
    
    all_results = runner.run_all_experiments(num_epochs=args.epochs)
    
    # Porovnanie vysledkov
    comparator = ResultsComparator(all_results)
    comparator.summarize()
    comparator.generate_latex_table(
        output_path=os.path.join("results", "latex_table.tex")
    )
    
    return all_results


def main():
    """
    Hlavna funkcia - vstupny bod programu.
    """
    # Vypis uvod
    print("=" * 70)
    print("  DETEKCIA PARKINSONOVEJ CHOROBY Z RECI")
    print("  Porovnanie Domain Adaptation Technik")
    print("  Autori: Dmytro Protsun, Mykyta Olym")
    print("  TUKE 2025/2026")
    print("=" * 70)
    
    # Parsujeme argumenty
    args = parse_arguments()
    
    # Nastavime seed
    set_all_seeds(args.seed)
    
    # Zistime zariadenie
    device = get_device()
    
    # Zaciname merit cas
    start_time = time.time()
    
    # Co chceme spustit?
    if args.quick:
        # Rychly test
        results = run_quick_test()
    elif args.run_all:
        # Vsetky experimenty
        results = run_all(args)
    elif args.method == "all":
        # Vsetky DA metody pre dany model a domeny
        runner = ExperimentRunner(
            model_types=[args.model],
            da_methods=ADAPTATION_METHODS,
        )
        results = runner.run_all_experiments(num_epochs=args.epochs)
    else:
        # Jeden konkretny experiment
        results = run_single(args)
    
    # Celkovy cas
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"  HOTOVO! Celkovy cas: {format_time(total_time)}")
    print(f"  Vysledky su v priecinku: results/")
    print(f"  Modely su v priecinku: saved_models/")
    print(f"{'='*70}")


# Toto sa spusti ked zavolame: python main.py
if __name__ == "__main__":
    main()
