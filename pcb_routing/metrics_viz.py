import csv
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_rows(files: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["routed"] = int(row.get("routed", 0))
                except Exception:
                    row["routed"] = 0
                for k in ("total_length", "total_turns", "layers_used", "time_sec"):
                    try:
                        row[k] = float(row.get(k, 0))
                    except Exception:
                        row[k] = 0.0
                rows.append(row)
    return rows


def aggregate_by_algo_method(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r.get("algo", "")), str(r.get("method", "")))
        groups.setdefault(key, []).append(r)
    out = []
    for (algo, method), items in groups.items():
        def col(name):
            return [float(x.get(name, 0)) for x in items]
        routed = [int(x.get("routed", 0)) for x in items]
        length = col("total_length")
        turns = col("total_turns")
        layers = col("layers_used")
        time_s = col("time_sec")
        out.append({
            "algo": algo,
            "method": method,
            "count": len(items),
            "routed_mean": mean(routed) if routed else 0,
            "routed_median": median(routed) if routed else 0,
            "length_mean": mean(length) if length else 0.0,
            "length_median": median(length) if length else 0.0,
            "turns_mean": mean(turns) if turns else 0.0,
            "turns_median": median(turns) if turns else 0.0,
            "layers_mean": mean(layers) if layers else 0.0,
            "layers_median": median(layers) if layers else 0.0,
            "time_mean": mean(time_s) if time_s else 0.0,
            "time_median": median(time_s) if time_s else 0.0,
        })
    out.sort(key=lambda d: (d["algo"], d["method"]))
    return out


def plot_summary_bar(csv_files: List[str], metric: str, use_median: bool, out_path: str, title: str = "Summary") -> None:
    rows = _read_rows(csv_files)
    summary = aggregate_by_algo_method(rows)
    # metric key
    stat_key = f"{metric}_{'median' if use_median else 'mean'}"
    labels = [f"{r['algo']}:{r['method']}" for r in summary]
    values = [float(r.get(stat_key, 0.0)) for r in summary]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
    bars = ax.bar(labels, values, color="#60a5fa")
    ax.set_title(title)
    ax.set_ylabel(stat_key.replace('_', ' '))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    # annotate
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0,3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

