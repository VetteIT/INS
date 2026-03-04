"""
Evaluacny modul - vyhodnocuje modely pomocou roznych metrik.
Tu spocitame vsetky metriky ktore potrebujeme v sprave:
- Accuracy, AUC, F1-score, Senzitivita, Specificita

A tiez tu porovnavame in-domain vs out-of-domain vykonnost.

Autori: Dmytro Protsun, Mykyta Olym
"""

import numpy as np

from utils.metrics import compute_all_metrics


class ModelEvaluator:
    """
    Trieda na evaluaciu modelov.
    Vie spocitat vsetky relevantne metriky a porovnat
    in-domain vs out-of-domain vykonnost.
    """

    def __init__(self):
        """Inicializacia evaluatora."""
        # Tu si budeme ukladat vysledky vsetkych experimentov
        self.all_results = {}

    def evaluate_model(self, trainer, test_loader, experiment_name):
        """
        Vyhodnoti model a ulozi vysledky.
        
        Parametre:
            trainer: ModelTrainer objekt (alebo DA trainer)
            test_loader: DataLoader s testovacimi datami
            experiment_name (str): nazov experimentu
        
        Vrati:
            dict: vysledky evaluacie so vsetkymi metrikami
        """
        print(f"\n--- Evaluacia: {experiment_name} ---")
        
        # Ziskame predikcie
        raw_results = trainer.evaluate(test_loader)
        
        # Spocitame vsetky metriky
        metrics = compute_all_metrics(
            y_true=raw_results["labels"],
            y_pred=raw_results["predictions"],
            y_prob=raw_results["probabilities"],
        )
        
        # Pridame surove data
        metrics["raw_predictions"] = raw_results["predictions"]
        metrics["raw_labels"] = raw_results["labels"]
        metrics["raw_probabilities"] = raw_results["probabilities"]
        
        # Ulozime vysledky
        self.all_results[experiment_name] = metrics
        
        # Vypiseme vysledky
        self._print_results(experiment_name, metrics)
        
        return metrics

    def _print_results(self, experiment_name, metrics):
        """
        Vypiese vysledky do konzoly.
        Pekne formatovane aby sa dobre citali.
        """
        print(f"\n  Vysledky pre: {experiment_name}")
        print(f"  {'='*40}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  AUC:          {metrics['auc']:.4f}")
        print(f"  F1-score:     {metrics['f1_score']:.4f}")
        print(f"  Senzitivita:  {metrics['sensitivity']:.4f}")
        print(f"  Specificita:  {metrics['specificity']:.4f}")
        print(f"  {'='*40}")

    def compare_in_out_domain(self, in_domain_name, out_domain_name):
        """
        Porovná in-domain a out-of-domain výkonnosť.
        Ukazuje ako veľmi sa zhorsí model ked ho testujeme na inej doméne.
        
        Parametre:
            in_domain_name: nazov in-domain experimentu
            out_domain_name: nazov out-of-domain experimentu
        
        Vrati:
            dict: rozdiely medzi in-domain a out-of-domain
        """
        if in_domain_name not in self.all_results:
            print(f"  VAROVANIE: Experiment '{in_domain_name}' nenajdeny!")
            return None
        if out_domain_name not in self.all_results:
            print(f"  VAROVANIE: Experiment '{out_domain_name}' nenajdeny!")
            return None
        
        in_metrics = self.all_results[in_domain_name]
        out_metrics = self.all_results[out_domain_name]
        
        print(f"\n  Porovnanie In-Domain vs Out-of-Domain:")
        print(f"  {'Metrika':<15} {'In-Domain':>12} {'Out-Domain':>12} {'Rozdiel':>12}")
        print(f"  {'-'*51}")
        
        differences = {}
        for metric_name in ["accuracy", "auc", "f1_score", "sensitivity", "specificity"]:
            in_val = in_metrics[metric_name]
            out_val = out_metrics[metric_name]
            diff = out_val - in_val
            differences[metric_name] = diff
            
            # Formatovany vypis
            arrow = "↓" if diff < 0 else "↑" if diff > 0 else "="
            print(f"  {metric_name:<15} {in_val:>12.4f} {out_val:>12.4f} {diff:>+12.4f} {arrow}")
        
        return differences

    def compare_da_methods(self, method_names):
        """
        Porovná viacero DA techník medzi sebou.
        
        Parametre:
            method_names (list): nazvy experimentov na porovnanie
        
        Vrati:
            dict: tabulka vysledkov
        """
        print(f"\n{'='*70}")
        print(f"  Porovnanie Domain Adaptation technik")
        print(f"{'='*70}")
        
        # Hlavicka
        header = f"  {'Metoda':<25}"
        for metric in ["Accuracy", "AUC", "F1", "Sens.", "Spec."]:
            header += f" {metric:>8}"
        print(header)
        print(f"  {'-'*65}")
        
        comparison = {}
        
        for method_name in method_names:
            if method_name not in self.all_results:
                print(f"  {method_name:<25} -- vysledky nenajdene --")
                continue
            
            metrics = self.all_results[method_name]
            comparison[method_name] = metrics
            
            row = f"  {method_name:<25}"
            row += f" {metrics['accuracy']:>8.4f}"
            row += f" {metrics['auc']:>8.4f}"
            row += f" {metrics['f1_score']:>8.4f}"
            row += f" {metrics['sensitivity']:>8.4f}"
            row += f" {metrics['specificity']:>8.4f}"
            print(row)
        
        print(f"  {'-'*65}")
        
        # Najdeme najlepsiu metodu
        if comparison:
            best_method = max(comparison.keys(),
                            key=lambda m: comparison[m]["f1_score"])
            print(f"\n  Najlepsia metoda (podla F1): {best_method}")
            print(f"  F1-score: {comparison[best_method]['f1_score']:.4f}")
        
        return comparison

    def get_all_results(self):
        """Vrati vsetky vysledky."""
        return self.all_results

    def generate_report(self, output_path=None):
        """
        Vygeneruje textovu spravu so vsetkymi vysledkami.
        
        Parametre:
            output_path: cesta kde sa ulozi sprava
        """
        import os
        from config.settings import RESULTS_DIR
        
        if output_path is None:
            output_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        lines = []
        lines.append("=" * 70)
        lines.append("EVALUATION REPORT")
        lines.append("Porovnanie domain adaptation technik pre detekciu PD")
        lines.append("Autori: Dmytro Protsun, Mykyta Olym")
        lines.append("=" * 70)
        lines.append("")
        
        for exp_name, metrics in self.all_results.items():
            lines.append(f"Experiment: {exp_name}")
            lines.append(f"  Accuracy:     {metrics['accuracy']:.4f}")
            lines.append(f"  AUC:          {metrics['auc']:.4f}")
            lines.append(f"  F1-score:     {metrics['f1_score']:.4f}")
            lines.append(f"  Senzitivita:  {metrics['sensitivity']:.4f}")
            lines.append(f"  Specificita:  {metrics['specificity']:.4f}")
            lines.append("")
        
        report_text = "\n".join(lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        print(f"\n  Sprava ulozena: {output_path}")
        
        return report_text
