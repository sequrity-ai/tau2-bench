"""
Plot PPL histograms (with and without data) for each perplexity ratio result file.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_FILES = {
    "retail": "results/ppl_retail.json",
    "telecom": "results/ppl_telecom.json",
    "telecom-workflow": "results/ppl_telecom_workflow.json",
}

OUTPUT_DIR = Path("results")


def load_ppl_values(path: str) -> tuple[list[float], list[float]]:
    with open(path) as f:
        data = json.load(f)
    ppl_without = []
    ppl_with = []
    for task in data.get("per_task", []):
        if "ppl_without_data" in task and "ppl_with_data" in task:
            ppl_without.append(task["ppl_without_data"])
            ppl_with.append(task["ppl_with_data"])
    return ppl_without, ppl_with


def plot_ppl_histogram(ppl_without: list[float], ppl_with: list[float], domain: str, output_path: Path):
    all_values = ppl_without + ppl_with
    bin_min = min(all_values)
    bin_max = max(all_values)
    bins = np.linspace(bin_min, bin_max, 20)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(ppl_without, bins=bins, alpha=0.6, color="cornflowerblue", label="ppl_without_data")
    ax.hist(ppl_with, bins=bins, alpha=0.6, color="lightcoral", label="ppl_with_data")

    ax.set_title(f"Histogram of PPL Values With and Without Data ({domain})")
    ax.set_xlabel("PPL Value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, result_file in RESULT_FILES.items():
        path = Path(result_file)
        if not path.exists():
            print(f"Skipping {domain}: {path} not found")
            continue

        ppl_without, ppl_with = load_ppl_values(path)
        if not ppl_without:
            print(f"Skipping {domain}: no valid PPL data")
            continue

        output_path = OUTPUT_DIR / f"ppl_histogram_{domain}.png"
        plot_ppl_histogram(ppl_without, ppl_with, domain, output_path)


if __name__ == "__main__":
    main()
