"""
Modul na porovnanie a analyzu vysledkov experimentov.
Tu analyzujeme ktora kombinacia modelu a DA techniky je najlepsia.

Autori: Dmytro Protsun, Mykyta Olym
"""

import os
import numpy as np
import pandas as pd

from config.settings import RESULTS_DIR


class ResultsComparator:
    """
    Trieda na porovnanie vysledkov z roznych experimentov.
    Vie nacitat ulozené vysledky a analyzovat ich.
    """

    def __init__(self, results_dict=None):
        """
        Parametre:
            results_dict: slovnik s vysledkami (ak uz mame v pameti)
        """
        self.results = results_dict or {}

    def load_results_from_csv(self, csv_path=None):
        """
        Nacita vysledky z CSV suboru.
        
        Parametre:
            csv_path: cesta k CSV suboru
        """
        if csv_path is None:
            csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
        
        if not os.path.exists(csv_path):
            print(f"  CSV subor neexistuje: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            exp_name = row.get("Experiment", "unknown")
            self.results[exp_name] = {
                col: float(row[col]) if col != "Experiment" else row[col]
                for col in df.columns
            }
        
        print(f"  Nacitanych {len(self.results)} experimentov z {csv_path}")

    def find_best_method(self, metric="f1_score"):
        """
        Najde najlepsiu kombinaciu modelu a DA techniky.
        
        Parametre:
            metric (str): metrika podla ktorej porovnavame
        
        Vrati:
            str: nazov najlepsieho experimentu
        """
        if not self.results:
            print("  Ziadne vysledky na porovnanie!")
            return None
        
        best_name = None
        best_value = -1
        
        for exp_name, metrics in self.results.items():
            value = metrics.get(metric, 0)
            if isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best_name = exp_name
        
        print(f"\n  Najlepsi experiment podla {metric}: {best_name}")
        print(f"  {metric} = {best_value:.4f}")
        
        return best_name

    def analyze_domain_shift(self):
        """
        Analyzuje pokles vykonu medzi in-domain a out-of-domain testovanim.
        Pre kazdu DA techniku spocita priemerny pokles.
        """
        print("\n" + "=" * 60)
        print("  ANALYZA DOMAIN SHIFT")
        print("=" * 60)
        
        # Roztriedime vysledky podla DA metody
        by_method = {}
        
        for exp_name, metrics in self.results.items():
            # Zistime ci je to in-domain alebo out-domain
            if "_in_domain" in exp_name:
                # Najdeme zodpovedajuci out-domain
                base_name = exp_name.replace("_in_domain", "")
                out_name = f"{base_name}_out_domain"
                
                if out_name in self.results:
                    # Zistime DA metodu z nazvu
                    parts = base_name.split("_")
                    if len(parts) >= 2:
                        da_method = parts[1]
                    else:
                        da_method = "unknown"
                    
                    if da_method not in by_method:
                        by_method[da_method] = {"drops": []}
                    
                    in_f1 = metrics.get("f1_score", 0)
                    out_f1 = self.results[out_name].get("f1_score", 0)
                    drop = in_f1 - out_f1
                    by_method[da_method]["drops"].append(drop)
        
        # Vypiseme priemerne poklesy
        print(f"\n  {'DA Metoda':<20} {'Priemy pokles F1':>18} {'Std':>10}")
        print(f"  {'-'*48}")
        
        for method, data in sorted(by_method.items()):
            drops = data["drops"]
            if drops:
                mean_drop = np.mean(drops)
                std_drop = np.std(drops)
                print(f"  {method:<20} {mean_drop:>+18.4f} {std_drop:>10.4f}")
        
        return by_method

    def generate_latex_table(self, output_path=None):
        """
        Vygeneruje LaTeX tabulku s vysledkami.
        Uzitocne pre semestrálnu pracu.
        
        Parametre:
            output_path: cesta na ulozenie .tex suboru
        """
        if not self.results:
            print("  Ziadne vysledky!")
            return ""
        
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Porovnanie domain adaptation techník}")
        lines.append("\\label{tab:results}")
        lines.append("\\begin{tabular}{lcccccc}")
        lines.append("\\hline")
        lines.append("Metóda & Accuracy & AUC & F1 & Sens. & Spec. & Čas (s) \\\\")
        lines.append("\\hline")
        
        for exp_name, metrics in sorted(self.results.items()):
            acc = metrics.get("accuracy", 0)
            auc = metrics.get("auc", 0)
            f1 = metrics.get("f1_score", 0)
            sens = metrics.get("sensitivity", 0)
            spec = metrics.get("specificity", 0)
            time_val = metrics.get("training_time", 0)
            
            if isinstance(acc, (int, float)):
                # Skratime nazov experimentu
                short_name = exp_name.replace("_", "\\_")
                if len(short_name) > 30:
                    short_name = short_name[:30] + "..."
                
                line = (f"{short_name} & "
                       f"{acc:.3f} & {auc:.3f} & {f1:.3f} & "
                       f"{sens:.3f} & {spec:.3f} & {time_val:.1f} \\\\")
                lines.append(line)
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        latex_code = "\n".join(lines)
        
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(latex_code)
            print(f"  LaTeX tabulka ulozena: {output_path}")
        
        return latex_code

    def summarize(self):
        """
        Vypise celkove zhrnutie vsetkych experimentov.
        """
        print("\n" + "=" * 70)
        print("  CELKOVE ZHRNUTIE EXPERIMENTOV")
        print("=" * 70)
        
        if not self.results:
            print("  Ziadne vysledky na zhrnutie!")
            return
        
        # Najlepsia metoda podla roznych metrik
        for metric in ["accuracy", "auc", "f1_score", "sensitivity"]:
            best = self.find_best_method(metric)
        
        # Domain shift analyza
        self.analyze_domain_shift()
        
        # Celkove zaver
        print("\n  ZAVER:")
        print("  " + "-" * 40)
        
        best_overall = self.find_best_method("f1_score")
        if best_overall:
            best_metrics = self.results[best_overall]
            print(f"  Najlepsia celkova metoda: {best_overall}")
            print(f"  F1 = {best_metrics.get('f1_score', 0):.4f}")
            print(f"  AUC = {best_metrics.get('auc', 0):.4f}")
        
        print("\n  Pre realne klinické nasadenie odporucame")
        print("  metodu s najvyssou senzitivitou (aby sme")
        print("  neprehliadli PD pacienta).")
