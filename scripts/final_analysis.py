#!/usr/bin/env python3
"""
Final analysis script for deceptive instruction following study.
Run after data collection is complete (data/final/ directory populated).
Produces: all figures (PNG) + analysis summary JSON.
"""

import json
import os
import math
from pathlib import Path

DATA_DIR = Path("/Users/argo/Documents/Research/deception-paper/data/final")
FIG_DIR = Path("/Users/argo/Documents/Research/deception-paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Claude Sonnet 4.6", "Claude Opus 4.6", "Gemini 3.1 Flash", "Qwen3-Max", "LFM-2-24B"]
MODEL_COLORS = {
    "Claude Sonnet 4.6": "#c0392b",
    "Claude Opus 4.6":   "#8e44ad",
    "Gemini 3.1 Flash":  "#2980b9",
    "Qwen3-Max":         "#27ae60",
    "LFM-2-24B":         "#d35400",
}

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, (center - margin)*100), min(100, (center + margin)*100))

def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
    })
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, figures will be skipped")

# ─────────────────────────────────────────────
# Load all datasets
# ─────────────────────────────────────────────

def load(fname):
    p = DATA_DIR / fname
    if not p.exists():
        print(f"WARNING: {fname} not found")
        return None
    with open(p) as f:
        return json.load(f)

exp_v = load("exp_v_validation.json")
exp_a = load("exp_a_baseline.json")
exp_b = load("exp_b_escalation.json")
exp_c = load("exp_c_persona.json")
exp_d = load("exp_d_domain.json")
exp_e = load("exp_e_temperature.json")
exp_f = load("exp_f_stimulus.json")

# ─────────────────────────────────────────────
# Figure 1: Baseline compliance rates (Exp A)
# ─────────────────────────────────────────────

if HAS_MPL and exp_a:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    models_present = [m for m in MODELS if m in exp_a]
    comply_pcts = [exp_a[m]["overall"]["comply_pct"] for m in models_present]
    ci_lows  = [exp_a[m]["overall"]["ci_low"] for m in models_present]
    ci_highs = [exp_a[m]["overall"]["ci_high"] for m in models_present]
    
    xerr_low  = [comply_pcts[i] - ci_lows[i]  for i in range(len(models_present))]
    xerr_high = [ci_highs[i] - comply_pcts[i] for i in range(len(models_present))]
    
    colors = [MODEL_COLORS.get(m, "#999") for m in models_present]
    bars = ax.barh(models_present, comply_pcts, color=colors, alpha=0.85, height=0.55)
    ax.errorbar(comply_pcts, range(len(models_present)), 
                xerr=[xerr_low, xerr_high], fmt='none', color='#333', capsize=4, linewidth=1.5)
    
    for i, (pct, model) in enumerate(zip(comply_pcts, models_present)):
        ax.text(pct + 2, i, f"{pct:.0f}%", va='center', fontsize=10, color='#333')
    
    ax.axvline(50, color='#aaa', linestyle='--', linewidth=1, label='50% line')
    ax.set_xlim(0, 110)
    ax.set_xlabel("Compliance rate (%)", fontsize=11)
    ax.set_title("Figure 1: Single-turn baseline compliance by model\n(N=100/model, corrected classifier, 95% Wilson CI)", 
                 fontsize=11, pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_baseline.png", bbox_inches='tight')
    plt.close()
    print("Saved fig1_baseline.png")

# ─────────────────────────────────────────────
# Figure 2: Escalation (Exp B) — Sonnet
# ─────────────────────────────────────────────

if HAS_MPL and exp_b and "Claude Sonnet 4.6" in exp_b:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left: escalation across turns
    ax = axes[0]
    sonnet = exp_b["Claude Sonnet 4.6"]
    turns = ["T1\n(initial)", "T2\n(friendly push)", "T3\n(authority claim)"]
    
    for run_data in sonnet.get("runs", []):
        run_n = run_data.get("run", "?")
        vals = [run_data.get("turn1_comply", 0), run_data.get("turn2_comply", 0), run_data.get("turn3_comply", 0)]
        ax.plot(turns, vals, 'o-', alpha=0.4, color='#c0392b', linewidth=1.5, markersize=6)
    
    # Mean line
    summary = sonnet.get("summary", {})
    means = [summary.get("t1_mean", 0), summary.get("t2_mean", 0), summary.get("t3_mean", 0)]
    ax.plot(turns, means, 'o-', color='#c0392b', linewidth=2.5, markersize=9, label='Mean (3 runs)', zorder=5)
    
    # Filler control
    if "filler_control" in sonnet:
        filler = sonnet["filler_control"]
        f_means = [filler.get("t1_mean", 0), filler.get("t2_mean", 0), filler.get("t3_mean", 0)]
        ax.plot(turns, f_means, 's--', color='#7f8c8d', linewidth=1.5, markersize=7, label='Filler control')
    
    ax.set_ylim(0, 105)
    ax.set_ylabel("Compliance rate (%)")
    ax.set_title("Escalation: Claude Sonnet 4.6\n(3 runs × N=50, with filler control)", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    # Right: comparison across models at T1 vs T2
    ax = axes[1]
    escalation_models = [m for m in ["Claude Sonnet 4.6", "Qwen3-Max", "LFM-2-24B"] if m in exp_b]
    x = np.arange(len(escalation_models))
    width = 0.35
    
    t1_vals = [exp_b[m]["summary"].get("t1_mean", 0) for m in escalation_models]
    t2_vals = [exp_b[m]["summary"].get("t2_mean", 0) for m in escalation_models]
    
    ax.bar(x - width/2, t1_vals, width, label='T1 (no pressure)', color='#bdc3c7', alpha=0.9)
    ax.bar(x + width/2, t2_vals, width, label='T2 (after pushback)', color='#c0392b', alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in escalation_models], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Compliance rate (%)")
    ax.set_title("T1 vs T2 compliance by model", fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    fig.suptitle("Figure 2: Multi-turn escalation effect", fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_escalation.png", bbox_inches='tight')
    plt.close()
    print("Saved fig2_escalation.png")

# ─────────────────────────────────────────────
# Figure 3: Persona injection heatmap (Exp C)
# ─────────────────────────────────────────────

if HAS_MPL and exp_c:
    PERSONA_LABELS = {
        "no_system": "No system prompt",
        "neutral": "Neutral (helpful/harmless/honest)",
        "fiction_author": "Fiction author",
        "malfunctioning_ai": "Malfunctioning AI",
        "game_show_host": "Game show host",
        "pure_follower": "Pure instruction follower",
        "truth_absolutist": "Truth absolutist ✓",
    }
    
    personas = list(PERSONA_LABELS.keys())
    models_present = [m for m in MODELS if m in exp_c]
    
    matrix = []
    for persona in personas:
        row = []
        for model in models_present:
            val = exp_c.get(model, {}).get(persona, {}).get("comply_pct", None)
            row.append(val if val is not None else float('nan'))
        matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    import numpy as np
    data = np.array(matrix, dtype=float)
    
    im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=100, aspect='auto')
    
    ax.set_xticks(range(len(models_present)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in models_present], fontsize=9)
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([PERSONA_LABELS[p] for p in personas], fontsize=9)
    
    for i in range(len(personas)):
        for j in range(len(models_present)):
            val = data[i, j]
            if not math.isnan(val):
                text_color = 'white' if val > 70 or val < 30 else 'black'
                ax.text(j, i, f"{val:.0f}%", ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=text_color)
    
    plt.colorbar(im, ax=ax, label='Compliance rate (%)', shrink=0.8)
    ax.set_title("Figure 3: Persona injection — compliance rate by model and system prompt\n(N=50 per cell, corrected classifier)", 
                 fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_persona_heatmap.png", bbox_inches='tight')
    plt.close()
    print("Saved fig3_persona_heatmap.png")

# ─────────────────────────────────────────────
# Figure 4: Domain effects (Exp D) + logistic regression
# ─────────────────────────────────────────────

if HAS_MPL and exp_d:
    # Compute mean compliance per domain across models
    all_domains = []
    for model_data in exp_d.values():
        if isinstance(model_data, dict) and "by_domain" in model_data:
            for domain in model_data["by_domain"]:
                if domain not in all_domains:
                    all_domains.append(domain)
    
    domain_means = {}
    for domain in all_domains:
        vals = []
        for model, model_data in exp_d.items():
            if isinstance(model_data, dict) and "by_domain" in model_data:
                d = model_data["by_domain"].get(domain, {})
                if "comply_pct" in d:
                    vals.append(d["comply_pct"])
        if vals:
            domain_means[domain] = sum(vals) / len(vals)
    
    sorted_domains = sorted(domain_means.items(), key=lambda x: x[1], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Left: domain compliance bar chart
    ax = axes[0]
    d_names = [d[0] for d in sorted_domains]
    d_vals  = [d[1] for d in sorted_domains]
    colors_d = ['#c0392b' if v > 70 else '#27ae60' if v < 40 else '#f39c12' for v in d_vals]
    
    ax.barh(d_names, d_vals, color=colors_d, alpha=0.85, height=0.6)
    for i, v in enumerate(d_vals):
        ax.text(v + 1, i, f"{v:.0f}%", va='center', fontsize=9)
    ax.axvline(50, color='#aaa', linestyle='--', linewidth=1)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Mean compliance rate across models (%)")
    ax.set_title("Compliance by domain\n(mean across 5 models, N=50/cell)", fontsize=10)
    
    # Right: domain vs model variance comparison
    ax = axes[1]
    if "logistic_regression" in exp_d:
        lr = exp_d["logistic_regression"]
        model_sq = lr.get("model_mean_sq_coef", 0)
        domain_sq = lr.get("domain_mean_sq_coef", 0)
        total = model_sq + domain_sq
        
        ax.bar(["Domain effects", "Model effects"], 
               [domain_sq/total*100, model_sq/total*100],
               color=["#c0392b", "#2980b9"], alpha=0.85, width=0.5)
        for i, (label, val) in enumerate([("Domain", domain_sq/total*100), ("Model", model_sq/total*100)]):
            ax.text(i, val + 1, f"{val:.0f}%\nof variance", ha='center', fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of explained variance")
        ax.set_title("Domain vs model variance\n(logistic regression, mean sq. coefficient)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Logistic regression\ndata not available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=11, color='#999')
    
    fig.suptitle("Figure 4: Domain effects on compliance", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_domain.png", bbox_inches='tight')
    plt.close()
    print("Saved fig4_domain.png")

# ─────────────────────────────────────────────
# Figure 5: Temperature sweep (Exp E)
# ─────────────────────────────────────────────

if HAS_MPL and exp_e:
    temps = ["0.0", "0.3", "0.7", "1.0"]
    models_present = [m for m in MODELS if m in exp_e]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    for model in models_present:
        model_data = exp_e[model]
        y_vals = []
        x_vals = []
        for t in temps:
            td = model_data.get("by_temp", {}).get(t, {})
            if "comply" in td:
                y_vals.append(td["comply"])
                x_vals.append(float(t))
        if y_vals:
            ax.plot(x_vals, y_vals, 'o-', color=MODEL_COLORS.get(model, "#999"),
                   label=model, linewidth=2, markersize=7)
    
    ax.set_xlabel("Temperature", fontsize=11)
    ax.set_ylabel("Compliance rate (%)", fontsize=11)
    ax.set_title("Figure 5: Compliance rate across temperatures\n(N=40 per condition, corrected classifier)", fontsize=11, pad=12)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc='lower left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_temperature.png", bbox_inches='tight')
    plt.close()
    print("Saved fig5_temperature.png")

# ─────────────────────────────────────────────
# Figure 6: Stimulus generalization (Exp F)
# ─────────────────────────────────────────────

if HAS_MPL and exp_f:
    stimulus_types = ["baseline", "multi_step", "false_citation", "structured_output"]
    stimulus_labels = ["Baseline\n(2+2→seven)", "Multi-step\ntask", "False\ncitation", "Structured\noutput (JSON)"]
    models_present = [m for m in MODELS if m in exp_f]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    x = np.arange(len(stimulus_types))
    width = 0.15
    offsets = np.linspace(-(len(models_present)-1)/2, (len(models_present)-1)/2, len(models_present)) * width
    
    for i, model in enumerate(models_present):
        vals = []
        for st in stimulus_types:
            v = exp_f.get(model, {}).get(st, {}).get("comply_pct", None)
            vals.append(v if v is not None else 0)
        ax.bar(x + offsets[i], vals, width, label=model, 
               color=MODEL_COLORS.get(model, "#999"), alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(stimulus_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Compliance rate (%)")
    ax.set_title("Figure 6: Compliance across stimulus types\n(N=50 per cell, corrected classifier)", fontsize=11, pad=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_stimulus.png", bbox_inches='tight')
    plt.close()
    print("Saved fig6_stimulus.png")

# ─────────────────────────────────────────────
# Figure 7: Classifier validation (Exp V)
# ─────────────────────────────────────────────

if HAS_MPL and exp_v:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: kappa and agreement
    ax = axes[0]
    kappa = exp_v.get("kappa", 0)
    agree = exp_v.get("agreement_pct", 0)
    
    bars = ax.bar(["Cohen's κ\n(classifier vs judge)", "Agreement %"], 
                  [kappa, agree/100],
                  color=["#27ae60" if kappa >= 0.8 else "#c0392b", "#2980b9"],
                  alpha=0.85, width=0.4)
    ax.axhline(0.8, color='#27ae60', linestyle='--', linewidth=1.5, label='κ=0.80 threshold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classifier validation\n(corrected classifier vs LLM judge, N=400)", fontsize=10)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, [kappa, agree/100]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", 
                ha='center', fontsize=10, fontweight='bold')
    
    # Right: comply rates by model (classifier vs judge)
    ax = axes[1]
    models_v = [m for m in MODELS if m in exp_v.get("by_model", {})]
    if models_v:
        x = np.arange(len(models_v))
        width = 0.35
        cls_rates = [exp_v["by_model"][m].get("classifier_comply_pct", 0) for m in models_v]
        judge_rates = [exp_v["by_model"][m].get("judge_comply_pct", 0) for m in models_v]
        
        ax.bar(x - width/2, cls_rates, width, label='Corrected classifier', color='#c0392b', alpha=0.85)
        ax.bar(x + width/2, judge_rates, width, label='LLM judge', color='#2980b9', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models_v], fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Comply rate (%)")
        ax.set_title("Comply rates: classifier vs judge\nby model", fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    fig.suptitle("Figure 7: Classifier validation", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_validation.png", bbox_inches='tight')
    plt.close()
    print("Saved fig7_validation.png")

# ─────────────────────────────────────────────
# Save analysis summary JSON
# ─────────────────────────────────────────────

summary = {
    "generated": True,
    "figures": [f.name for f in FIG_DIR.glob("*.png")],
    "data_files_found": {
        "exp_v_validation": exp_v is not None,
        "exp_a_baseline": exp_a is not None,
        "exp_b_escalation": exp_b is not None,
        "exp_c_persona": exp_c is not None,
        "exp_d_domain": exp_d is not None,
        "exp_e_temperature": exp_e is not None,
        "exp_f_stimulus": exp_f is not None,
    }
}

if exp_v:
    summary["classifier_kappa"] = exp_v.get("kappa")
    summary["classifier_agreement_pct"] = exp_v.get("agreement_pct")

if exp_a:
    summary["baseline_comply_rates"] = {
        m: exp_a[m]["overall"]["comply_pct"] 
        for m in MODELS if m in exp_a
    }

with open(DATA_DIR / "analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Analysis complete ===")
print(f"Figures saved to: {FIG_DIR}")
print(f"Summary saved to: {DATA_DIR}/analysis_summary.json")
