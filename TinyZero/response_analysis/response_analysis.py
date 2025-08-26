import math
import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
OUT_DIR = "/om/user/tiffany8/grpo-urop/TinyZero/response_analysis/"

def summarize_lengths(df, model_names):
    """Return a summary DataFrame with mean, std, min, max per model."""
    rows = []
    for m in model_names:
        col = "{}_len".format(m)
        vals = df[col].astype(int)
        rows.append({
            "model": m,
            "count": int(vals.count()),
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "min": int(vals.min()) if len(vals) else 0,
            "max": int(vals.max()) if len(vals) else 0,
        })
    summary = pd.DataFrame(rows).set_index("model").sort_index()
    return summary

def plot_length_histograms(df, model_names, df_name, bins=30):
    """Plot one histogram per model and save as PNGs."""
    for m in model_names:
        col = "{}_len".format(m)
        vals = df[col].astype(int).values

        plt.figure(figsize=(7,4.5))
        plt.hist(vals, bins=bins)
        plt.title("Response Length Distribution - {}".format(m))
        plt.xlabel("Characters")
        plt.ylabel("Count")
        plt.tight_layout()
        path = os.path.join(OUT_DIR + df_name, m + "_length_hist_" + df_name + ".png")
        plt.savefig(path, dpi=160)
        plt.close()

def plot_overlay_length_histograms(df, model_names, df_name, group_name, bins=30):
    """Plot overlay plot across models in model_names and save as PNGs."""
    # Overlay plot 
    plt.figure(figsize=(8,5))
    for m in model_names:
        col = "{}_len".format(m)
        vals = df[col].astype(int).values
        plt.hist(vals, bins=bins, histtype="step", label=m)
    plt.title("Response Length Distributions - " + group_name)
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR + df_name, group_name + "_length_hist_overlay_" + df_name + ".png")
    plt.savefig(path, dpi=160)
    plt.close()

def analyze(df, model_names, model_groups, df_name):
    """
    Run full analysis:
      - summarize
      - write CSVs
      - write plots
    Returns paths to outputs.
    """
    # Save the per-prompt lengths table
    per_prompt_csv = os.path.join(OUT_DIR + df_name, "per_prompt_lengths_" + df_name + ".csv")
    keep_cols = ["prompt"] + ["{}_len".format(m) for m in model_names if "{}_len".format(m) in df.columns]
    df.loc[:, keep_cols].to_csv(per_prompt_csv, index=False)

    # Summary stats
    summary = summarize_lengths(df, model_names)
    summary_csv = os.path.join(OUT_DIR + df_name, "length_summary_" + df_name + ".csv")
    summary.to_csv(summary_csv)

    # Plots
    plot_length_histograms(df, model_names, df_name, bins=30)
    for group_name, model_names in model_groups.items():
        plot_overlay_length_histograms(df, model_names, df_name, group_name, bins=30)

# --- Example usage ---
if __name__ == "__main__":
    # Example DataFrame
    df_name = 'mathqa'
    data_path = '/om/user/tiffany8/grpo-urop/TinyZero/response_analysis/' + df_name + '/responses_' + df_name + '.jsonl'
    model_names = ["llama8b base", "llama8b taco", "deepmath base", "deepmath taco", "mistral base", "mistral taco",
                    "qwen base", "qwen taco", "qwen 500",
                    "llama base", "llama taco", "llama 100", "llama 300", "llama 700"]
    model_groups = {'llama8b': ["llama8b base", "llama8b taco"], 
                    'deepmath': ["deepmath base", "deepmath taco"],
                    'mistral': ["mistral base", "mistral taco"],
                    'qwen': ["qwen base", "qwen taco", "qwen 500"],
                    'llama': ["llama base", "llama taco", "llama 100", "llama 300", "llama 700"],}
    df = pd.read_json(data_path, lines=True)

    os.makedirs(OUT_DIR + df_name, exist_ok=True)
    analyze(df, model_names, model_groups, df_name)
    print('wrote outputs to:', OUT_DIR + df_name)
