"""
Hlavny skript na spustenie VSETKYCH experimentov.
Spusta vsetky kombinacie:
- 3 typy modelov (CNN, Traditional, Wav2Vec2)
- 5 DA technik (Baseline, DANN, MMD, Contrastive, Multi-source)
- 3 domeny (PC-GITA, Neurovoz, PDITA) v roznych source/target kombinaciach

Celkovo to je vela experimentov, ale kazdy experiment sa spusti automaticky.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

from config.settings import (
    DOMAIN_NAMES, ADAPTATION_METHODS, MODEL_TYPES,
    RESULTS_DIR
)
from data.data_loader import get_domain_datasets, create_data_loaders
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from utils.helpers import (
    set_all_seeds, get_device, print_experiment_header,
    format_time, save_results_to_csv
)
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_training_history,
    plot_da_comparison
)


class ExperimentRunner:
    """
    Hlavna trieda na spustenie a riadenie experimentov.
    Obsahuje vsetku logiku na:
    - Nacitanie dat
    - Vytvorenie modelov
    - Spustenie trenovania
    - Evaluaciu
    - Ukladanie vysledkov
    """

    def __init__(self, model_types=None, da_methods=None, domains=None, device=None):
        """
        Parametre:
            model_types: list typov modelov na testovanie
            da_methods: list DA technik na testovanie
            domains: list nazov domen
            device: zariadenie (cpu/cuda)
        """
        self.model_types = model_types or MODEL_TYPES
        self.da_methods = da_methods or ADAPTATION_METHODS
        self.domains = domains or DOMAIN_NAMES
        self.device = device or get_device()
        
        # Evaluator na zbieranie vysledkov
        self.evaluator = ModelEvaluator()
        
        # Slovnik so vsetkymi vysledkami
        self.all_results = {}
        
        # Casomer
        self.start_time = None
        
        print(f"\nExperiment Runner inicializovany:")
        print(f"  Modely:     {self.model_types}")
        print(f"  DA metody:  {self.da_methods}")
        print(f"  Domeny:     {self.domains}")
        print(f"  Zariadenie: {self.device}")

    def run_single_experiment(self, model_type, da_method, source_domain,
                               target_domain, num_epochs=None):
        """
        Spusti jeden experiment s danou konfiguraciou.
        
        Parametre:
            model_type: typ modelu
            da_method: DA technika
            source_domain: zdrojova domena
            target_domain: cielova domena
            num_epochs: pocet epoch
        
        Vrati:
            dict: vysledky experimentu
        """
        experiment_name = f"{model_type}_{da_method}_{source_domain}_to_{target_domain}"
        
        print_experiment_header(
            experiment_name, model_type, da_method,
            source_domain, target_domain
        )
        
        set_all_seeds()
        
        try:
            # Zistime aky typ features potrebujeme pre dany model
            if model_type == "cnn":
                feature_type = "spectrogram"
            elif model_type == "traditional":
                feature_type = "features"
            else:
                feature_type = "raw"  # wav2vec berie surovy audio signal

            # 1. Nacitame datasety
            print("\n  Nacitavam datasety...")
            source_dataset = get_domain_datasets(source_domain, feature_type=feature_type)
            target_dataset = get_domain_datasets(target_domain, feature_type=feature_type)
            
            # Skontrolujeme ci mame data
            if len(source_dataset) == 0:
                print(f"  PRESKAKUJEM: Source dataset {source_domain} je prazdny!")
                return None
            if len(target_dataset) == 0:
                print(f"  PRESKAKUJEM: Target dataset {target_domain} je prazdny!")
                return None
            
            # 2. Vytvorime data loadery
            print("  Vytváram data loadery...")
            
            if da_method == "baseline":
                # Pre baseline trenujeme aj testujeme na source, potom testujeme na target
                source_train_loader, source_test_loader = create_data_loaders(source_dataset)
                target_train_loader, target_test_loader = create_data_loaders(target_dataset)
            else:
                # Pre DA metody potrebujeme aj source aj target loader
                source_train_loader, source_test_loader = create_data_loaders(source_dataset)
                target_train_loader, target_test_loader = create_data_loaders(target_dataset)
            
            # 3. Vytvorime trainer
            print(f"  Vytváram {model_type} model s {da_method}...")
            trainer = ModelTrainer(
                model_type=model_type,
                adaptation_method=da_method,
                device=str(self.device),
            )
            
            # 4. Trenovanie
            print("  Zacinam trenovanie...")
            start = time.time()
            
            if da_method == "baseline":
                history = trainer.train_baseline(
                    source_train_loader,
                    val_loader=source_test_loader,
                    num_epochs=num_epochs,
                )
            else:
                history = trainer.train_with_da(
                    source_train_loader,
                    target_train_loader,
                    val_loader=target_test_loader,
                    num_epochs=num_epochs,
                )
            
            train_time = time.time() - start
            print(f"  Trenovanie dokoncene za {format_time(train_time)}")
            
            # 5. Evaluacia na Source domene (in-domain)
            print("\n  Evaluacia na SOURCE domene (in-domain)...")
            in_domain_results = self.evaluator.evaluate_model(
                trainer,
                source_test_loader,
                f"{experiment_name}_in_domain",
            )
            
            # 6. Evaluacia na Target domene (out-of-domain)
            print("  Evaluacia na TARGET domene (out-of-domain)...")
            out_domain_results = self.evaluator.evaluate_model(
                trainer,
                target_test_loader,
                f"{experiment_name}_out_domain",
            )
            
            # 7. Ulozime vysledky
            results = {
                "experiment_name": experiment_name,
                "model_type": model_type,
                "da_method": da_method,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "training_time": train_time,
                "in_domain": in_domain_results,
                "out_domain": out_domain_results,
                "history": {k: v for k, v in history.items()
                           if not isinstance(v, np.ndarray)},
            }
            
            self.all_results[experiment_name] = results
            
            # 8. Vizualizacia
            self._save_experiment_plots(results, history)
            
            # 9. Ulozime model
            trainer.save_model(f"{experiment_name}.pt")
            
            return results
            
        except Exception as e:
            print(f"\n  CHYBA v experimente {experiment_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_multi_source_experiment(self, model_type, source_domains,
                                     target_domain, num_epochs=None):
        """
        Spusti multi-source DA experiment.
        Trénuje na viacerych source domenach a testuje na target.
        
        Parametre:
            model_type: typ modelu
            source_domains: list zdrojovych domen
            target_domain: cielova domena
            num_epochs: pocet epoch
        """
        source_str = "+".join(source_domains)
        experiment_name = f"{model_type}_multi_source_{source_str}_to_{target_domain}"
        
        print_experiment_header(
            experiment_name, model_type, "multi_source",
            source_str, target_domain
        )
        
        set_all_seeds()
        
        try:
            # Zistime feature type pre model
            if model_type == "cnn":
                feature_type = "spectrogram"
            elif model_type == "traditional":
                feature_type = "features"
            else:
                feature_type = "raw"

            # 1. Nacitame datasety
            print("\n  Nacitavam datasety...")
            source_datasets = []
            source_loaders = []
            
            for domain in source_domains:
                ds = get_domain_datasets(domain, feature_type=feature_type)
                if len(ds) == 0:
                    print(f"  VAROVANIE: {domain} je prazdny, preskakujem")
                    continue
                source_datasets.append(ds)
                train_loader, _ = create_data_loaders(ds)
                source_loaders.append(train_loader)
            
            target_dataset = get_domain_datasets(target_domain, feature_type=feature_type)
            if len(target_dataset) == 0:
                print(f"  PRESKAKUJEM: Target {target_domain} je prazdny!")
                return None
            
            target_train_loader, target_test_loader = create_data_loaders(target_dataset)
            
            # 2. Vytvorime trainer
            trainer = ModelTrainer(
                model_type=model_type,
                adaptation_method="multi_source",
                device=str(self.device),
            )
            
            # 3. Trenovanie
            start = time.time()
            history = trainer.train_multi_source(
                source_loaders,
                target_train_loader,
                val_loader=target_test_loader,
                num_epochs=num_epochs,
            )
            train_time = time.time() - start
            print(f"  Trenovanie dokoncene za {format_time(train_time)}")
            
            # 4. Evaluacia na target domene
            out_domain_results = self.evaluator.evaluate_model(
                trainer,
                target_test_loader,
                f"{experiment_name}_out_domain",
            )
            
            results = {
                "experiment_name": experiment_name,
                "model_type": model_type,
                "da_method": "multi_source",
                "source_domain": source_str,
                "target_domain": target_domain,
                "training_time": train_time,
                "out_domain": out_domain_results,
                "history": {k: v for k, v in history.items()
                           if not isinstance(v, np.ndarray)},
            }
            
            self.all_results[experiment_name] = results
            trainer.save_model(f"{experiment_name}.pt")
            
            return results
            
        except Exception as e:
            print(f"\n  CHYBA: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(self, num_epochs=None):
        """
        Spusti VSETKY experimenty - vsetky kombinacie modelov, DA a domen.
        POZOR: Toto moze trvat velmi dlho!
        """
        self.start_time = time.time()
        
        print("\n" + "=" * 70)
        print("  SPUSTAM VSETKY EXPERIMENTY")
        print("=" * 70)
        
        total_experiments = 0
        
        # Pre kazdy model
        for model_type in self.model_types:
            # Pre kazdu DA techniku
            for da_method in self.da_methods:
                if da_method == "multi_source":
                    # Multi-source - spustame specialne
                    continue
                
                # Pre kazdu kombinaciu source -> target
                for source_domain in self.domains:
                    for target_domain in self.domains:
                        if source_domain == target_domain:
                            continue  # preskocime in-domain
                        
                        self.run_single_experiment(
                            model_type, da_method,
                            source_domain, target_domain,
                            num_epochs=num_epochs,
                        )
                        total_experiments += 1
            
            # Multi-source experimenty
            if "multi_source" in self.da_methods:
                for target_domain in self.domains:
                    source_domains = [d for d in self.domains if d != target_domain]
                    self.run_multi_source_experiment(
                        model_type, source_domains, target_domain,
                        num_epochs=num_epochs,
                    )
                    total_experiments += 1
        
        # Celkovy cas
        total_time = time.time() - self.start_time
        print(f"\n\nVsetky experimenty dokoncene!")
        print(f"Celkovy pocet experimentov: {total_experiments}")
        print(f"Celkovy cas: {format_time(total_time)}")
        
        # Ulozime vsetky vysledky
        self._save_all_results()
        
        return self.all_results

    def _save_experiment_plots(self, results, history):
        """Ulozi grafy pre jeden experiment."""
        exp_name = results["experiment_name"]
        exp_dir = os.path.join(RESULTS_DIR, "plots", exp_name)
        
        # Training history
        plot_training_history(
            history,
            title=f"Training: {exp_name}",
            save_path=os.path.join(exp_dir, "training_history.png"),
        )
        
        # Confusion matrix (out-of-domain)
        if "out_domain" in results and "confusion_matrix" in results["out_domain"]:
            plot_confusion_matrix(
                results["out_domain"]["confusion_matrix"],
                title=f"Confusion Matrix: {exp_name}",
                save_path=os.path.join(exp_dir, "confusion_matrix.png"),
            )
        
        # ROC krivka (out-of-domain)
        if "out_domain" in results and "roc_curve" in results["out_domain"]:
            plot_roc_curve(
                results["out_domain"].get("roc_curve"),
                results["out_domain"].get("auc", 0.5),
                title=f"ROC: {exp_name}",
                save_path=os.path.join(exp_dir, "roc_curve.png"),
            )

    def _save_all_results(self):
        """Ulozi vsetky vysledky do suborov."""
        # Ulozime CSV
        csv_results = {}
        for exp_name, exp_data in self.all_results.items():
            if "out_domain" in exp_data:
                csv_results[exp_name] = {
                    k: v for k, v in exp_data["out_domain"].items()
                    if isinstance(v, (int, float))
                }
                csv_results[exp_name]["training_time"] = exp_data["training_time"]
        
        save_results_to_csv(
            csv_results,
            os.path.join(RESULTS_DIR, "all_results.csv"),
        )
        
        # Ulozime porovnanie DA metod (graf)
        self._save_comparison_plots()
        
        # Vygenerujeme textovu spravu
        self.evaluator.generate_report()

    def _save_comparison_plots(self):
        """Vytvori porovnavacie grafy medzi DA metodami."""
        # Pre kazdy model typ vytvorime porovnanie DA metod
        for model_type in self.model_types:
            comparison_results = {}
            
            for exp_name, exp_data in self.all_results.items():
                if (exp_data.get("model_type") == model_type and
                    "out_domain" in exp_data):
                    
                    method_name = exp_data["da_method"]
                    if method_name not in comparison_results:
                        comparison_results[method_name] = exp_data["out_domain"]
            
            if comparison_results:
                for metric in ["accuracy", "auc", "f1_score"]:
                    plot_da_comparison(
                        comparison_results,
                        metric_name=metric,
                        title=f"{model_type}: {metric.replace('_', ' ').title()}",
                        save_path=os.path.join(
                            RESULTS_DIR, "plots",
                            f"comparison_{model_type}_{metric}.png"
                        ),
                    )
