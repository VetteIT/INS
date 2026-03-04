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
            # Vysledky mozu byt priamo v dict alebo vnorene v "out_domain"
            if isinstance(metrics, dict) and "out_domain" in metrics:
                value = metrics["out_domain"].get(metric, 0)
            else:
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
        
        Podporuje dva formaty vysledkov:
        - ExperimentRunner format: {exp_name: {"in_domain": {...}, "out_domain": {...}, "da_method": ...}}
        - Flat format z CSV: {exp_name_in_domain: {metrics}, exp_name_out_domain: {metrics}}
        """
        print("\n" + "=" * 60)
        print("  ANALYZA DOMAIN SHIFT")
        print("=" * 60)
        
        by_method = {}
        
        for exp_name, exp_data in self.results.items():
            # ExperimentRunner format - vysledky su vnorene
            if isinstance(exp_data, dict) and "in_domain" in exp_data and "out_domain" in exp_data:
                da_method = exp_data.get("da_method", "unknown")
                
                in_f1 = exp_data["in_domain"].get("f1_score", 0) if isinstance(exp_data["in_domain"], dict) else 0
                out_f1 = exp_data["out_domain"].get("f1_score", 0) if isinstance(exp_data["out_domain"], dict) else 0
                
                if da_method not in by_method:
                    by_method[da_method] = {"drops": [], "experiments": []}
                
                drop = in_f1 - out_f1
                by_method[da_method]["drops"].append(drop)
                by_method[da_method]["experiments"].append(exp_name)
            
            # Flat format: hladame _in_domain / _out_domain pary
            elif "_in_domain" in exp_name:
                base_name = exp_name.replace("_in_domain", "")
                out_name = f"{base_name}_out_domain"
                
                if out_name in self.results:
                    parts = base_name.split("_")
                    if len(parts) >= 3 and parts[1] == "multi" and len(parts) > 2 and parts[2] == "source":
                        da_method = "multi_source"
                    elif len(parts) >= 2:
                        da_method = parts[1]
                    else:
                        da_method = "unknown"
                    
                    if da_method not in by_method:
                        by_method[da_method] = {"drops": [], "experiments": []}
                    
                    in_f1 = exp_data.get("f1_score", 0) if isinstance(exp_data, dict) else 0
                    out_f1 = self.results[out_name].get("f1_score", 0)
                    drop = in_f1 - out_f1
                    by_method[da_method]["drops"].append(drop)
                    by_method[da_method]["experiments"].append(base_name)
        
        # Vypiseme priemerne poklesy
        if not by_method:
            print("\n  Ziadne in-domain/out-domain pary na analyzu.")
            return by_method
        
        print(f"\n  {'DA Metoda':<20} {'Priem. pokles F1':>18} {'Std':>10}")
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
        
        Podporuje ExperimentRunner format (vnorene in_domain/out_domain)
        aj flat format z CSV.
        
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
        
        for exp_name, exp_data in sorted(self.results.items()):
            if not isinstance(exp_data, dict):
                continue
            
            # Ziskame metriky - mozu byt vnorene v "out_domain" alebo priamo
            if "out_domain" in exp_data and isinstance(exp_data["out_domain"], dict):
                m = exp_data["out_domain"]
                time_val = exp_data.get("training_time", 0)
            else:
                m = exp_data
                time_val = m.get("training_time", 0)
            
            acc = m.get("accuracy", 0)
            auc = m.get("auc", 0)
            f1 = m.get("f1_score", 0)
            sens = m.get("sensitivity", 0)
            spec = m.get("specificity", 0)
            
            if not isinstance(acc, (int, float)):
                continue
            
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
            best_data = self.results[best_overall]
            # Ziskame metriky - mozu byt vnorene v "out_domain"
            if isinstance(best_data, dict) and "out_domain" in best_data:
                bm = best_data["out_domain"]
            else:
                bm = best_data
            print(f"  Najlepsia celkova metoda: {best_overall}")
            print(f"  F1 = {bm.get('f1_score', 0):.4f}")
            print(f"  AUC = {bm.get('auc', 0):.4f}")
        
        print("\n  Pre realne klinické nasadenie odporucame")
        print("  metodu s najvyssou senzitivitou (aby sme")
        print("  neprehliadli PD pacienta).")
