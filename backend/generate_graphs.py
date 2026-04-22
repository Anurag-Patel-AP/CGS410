"""
Offline Graph Generation Utility.
=================================

This script dynamically loads the processed .json and .csv metrics and renders 
high-resolution publication-ready PNG plots that match the online web dashboard.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    backend_dir = os.path.dirname(__file__)
    output_dir = os.path.join(backend_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set professional research style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    LANG_COLORS = {'en': '#3b82f6', 'hi': '#10b981', 'ja': '#ef4444', 'de': '#f59e0b'}
    LANG_LABELS = {'en': 'English', 'hi': 'Hindi', 'ja': 'Japanese', 'de': 'German'}
    
    # ---------------------------------------------------------
    # 1. Model UAS vs Sentence Length Chart
    # ---------------------------------------------------------
    results_path = os.path.join(backend_dir, "test_results.json")
    if os.path.exists(results_path):
        print("Generating UAS vs Sentence Length Chart...")
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        
        # Plot E(UAS|length) baseline
        max_len = 60
        x_baseline = np.arange(1, max_len + 1)
        y_baseline = (1.0 / x_baseline) * 100
        plt.plot(x_baseline, y_baseline, color="gray", linestyle="--", label="E(UAS|length)", linewidth=2, alpha=0.7)
        
        for lang in ['en', 'hi', 'ja', 'de']:
            lang_df = df[df['language'] == lang]
            if not lang_df.empty:
                # Group by length to get exact empirical average
                avg_uas = lang_df.groupby('length')['uas'].mean().reset_index()
                avg_uas = avg_uas[avg_uas['length'] <= max_len]
                plt.plot(avg_uas['length'], avg_uas['uas'], 
                         color=LANG_COLORS[lang], 
                         label=LANG_LABELS[lang], 
                         linewidth=2.5, alpha=0.9)
                         
        plt.title("Model UAS vs Sentence Length (Zero-shot Attn-to-Dep)", fontsize=14, pad=15)
        plt.xlabel("Sentence Length (tokens)", fontsize=12)
        plt.ylabel("UAS (%)", fontsize=12)
        plt.xlim(1, max_len)
        plt.ylim(0, 100)
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "uas_vs_length.png"), dpi=300)
        plt.close()
    else:
        print(f"File not found: {results_path}")
        
    # ---------------------------------------------------------
    # 2. Average UAS by Language Chart
    # ---------------------------------------------------------
    if os.path.exists(results_path):
        print("Generating Average UAS Chart...")
        # Re-use df from test_results.json loaded above
        avg_overall = df.groupby('language')['uas'].mean().reset_index()
        avg_overall['language_name'] = avg_overall['language'].map(LANG_LABELS)
        
        plt.figure(figsize=(8, 6))
        barplot = sns.barplot(data=avg_overall, x='language_name', y='uas', palette=[LANG_COLORS[l] for l in avg_overall['language']])
        
        plt.title("Model Average UAS by Language", fontsize=14, pad=15)
        plt.xlabel("Language", fontsize=12)
        plt.ylabel("Average UAS (%)", fontsize=12)
        plt.ylim(0, 100)
        
        # Add exact values on top of bars
        for index, row in avg_overall.iterrows():
            barplot.text(index, row['uas'] + 1, f"{row['uas']:.1f}%", color='black', ha="center", fontsize=11)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "avg_uas_by_lang.png"), dpi=300)
        plt.close()
        
    print(f"Graphs successfully saved to: {output_dir}")

if __name__ == "__main__":
    main()
