#!/usr/bin/env python3
"""Generate all figures for the paper from experiment results."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTDIR = "/Users/argo/.openclaw/workspace/research/v2/figures"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'truth_first': '#2ecc71',
    'follow_clean': '#e74c3c',
    'refused': '#3498db',
    'false_first': '#e67e22',
    'other': '#95a5a6',
    'truthful': '#2ecc71',
    'deceived': '#e74c3c',
}

# ============================================================
# FIG 1: Positional Control — the key confound test
# ============================================================
print("Generating Fig 1: Positional Control...")
d1 = json.load(open("/Users/argo/.openclaw/workspace/research/v2/results_exp1.json"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 1: Positional Control Experiment\nDoes prompt order drive output order?", fontsize=14, fontweight='bold')

for ax_idx, (model_key, label) in enumerate([("qwen3-max-thinking", "Qwen3-Max-Thinking"), ("llama-3.1-70b-instruct", "Llama 3.1 70B")]):
    ax = axes[ax_idx]
    if model_key not in d1:
        ax.text(0.5, 0.5, 'No data', ha='center')
        continue
    
    conds = d1[model_key]
    cond_labels = ['Question\nFirst', 'Instruction\nFirst', 'Interleaved', 'Control\n(no instruction)']
    cond_keys = ['question_first', 'instruction_first', 'interleaved', 'control_no_instruction']
    
    truth_vals = [conds[k]['pct_truth_first'] for k in cond_keys]
    false_vals = [conds[k]['pct_false_first'] for k in cond_keys]
    
    x = np.arange(len(cond_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, truth_vals, w, label='Truth appears first', color=COLORS['truth_first'], alpha=0.85)
    bars2 = ax.bar(x + w/2, false_vals, w, label='False answer appears first', color=COLORS['follow_clean'], alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=10)
    ax.set_ylabel('% of responses')
    ax.set_ylim(0, 110)
    ax.set_title(label)
    ax.legend(fontsize=9)
    
    for bar in bars1:
        h = bar.get_height()
        if h > 2: ax.text(bar.get_x() + bar.get_width()/2, h+1, f'{h:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 2: ax.text(bar.get_x() + bar.get_width()/2, h+1, f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig1_positional_control.png", bbox_inches='tight')
plt.close()
print("  Saved fig1_positional_control.png")

# ============================================================
# FIG 2: Fact Types Heatmap (Exp2)
# ============================================================
print("Generating Fig 2: Fact Types Heatmap...")
d2 = json.load(open("/Users/argo/.openclaw/workspace/research/v2/results_exp2.json"))

fact_types = ['arithmetic', 'geography', 'science_color', 'literature', 'biology',
              'astronomy', 'language', 'physics', 'history', 'animals']
models_2 = list(d2.keys())

# Build matrix: rows=models, cols=fact types, value=pct_truth_first
matrix = []
for m in models_2:
    row = []
    for f in fact_types:
        if f in d2[m]:
            row.append(d2[m][f]['pct_truth_first'])
        else:
            row.append(0)
    matrix.append(row)

matrix = np.array(matrix)
fact_labels = ['Arithmetic', 'Geography', 'Science\n(color)', 'Literature', 'Biology',
               'Astronomy', 'Language', 'Physics', 'History', 'Animals']
model_labels = [m.replace('-instruct','').replace('qwen','Qwen').replace('llama','Llama').replace('meta-llama/','') for m in models_2]

fig, ax = plt.subplots(figsize=(13, max(3, len(models_2)*1.2)))
im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

ax.set_xticks(np.arange(len(fact_labels)))
ax.set_xticklabels(fact_labels, rotation=30, ha='right', fontsize=10)
ax.set_yticks(np.arange(len(model_labels)))
ax.set_yticklabels(model_labels, fontsize=10)

for i in range(len(model_labels)):
    for j in range(len(fact_labels)):
        val = matrix[i, j]
        color = 'white' if val < 30 or val > 70 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='% Truth-First responses')
ax.set_title("Figure 2: Truth-First Rate by Fact Domain and Model\n(green=truth prioritized, red=false answer output)", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig2_fact_types_heatmap.png", bbox_inches='tight')
plt.close()
print("  Saved fig2_fact_types_heatmap.png")

# ============================================================
# FIG 3: Framing Effects (Exp3) - Qwen3-Max
# ============================================================
print("Generating Fig 3: Framing Effects...")
d3 = json.load(open("/Users/argo/.openclaw/workspace/research/v2/results_exp3.json"))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 3: Effect of Prompt Framing on Model Behavior", fontsize=14, fontweight='bold')

for ax_idx, (model_key, title) in enumerate([
    ("qwen3-max-thinking", "Qwen3-Max-Thinking"),
    ("qwen2.5-72b-instruct", "Qwen2.5-72B")
]):
    ax = axes[ax_idx]
    if model_key not in d3:
        ax.text(0.5, 0.5, 'No data', ha='center')
        continue
    
    framings = d3[model_key]
    labels = list(framings.keys())
    truth_vals = [framings[f]['pct_truth_first'] for f in labels]
    follow_vals = [framings[f]['pct_follow_clean'] for f in labels]
    refused_vals = [framings[f]['pct_refused'] for f in labels]
    other_vals = [100 - t - fo - r for t, fo, r in zip(truth_vals, follow_vals, refused_vals)]
    
    x = np.arange(len(labels))
    ax.barh(x, truth_vals, color=COLORS['truth_first'], alpha=0.85, label='Truth-first')
    ax.barh(x, follow_vals, left=truth_vals, color=COLORS['follow_clean'], alpha=0.85, label='Follow clean')
    ax.barh(x, refused_vals, left=[t+f for t,f in zip(truth_vals, follow_vals)], color=COLORS['refused'], alpha=0.85, label='Refused deception')
    
    clean_labels = [l.replace('_X','').replace('_',' ') for l in labels]
    ax.set_yticks(x)
    ax.set_yticklabels(clean_labels, fontsize=10)
    ax.set_xlabel('% of responses')
    ax.set_xlim(0, 105)
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig3_framing_effects.png", bbox_inches='tight')
plt.close()
print("  Saved fig3_framing_effects.png")

# ============================================================
# FIG 4: Instruction Hierarchy (Exp5)
# ============================================================
print("Generating Fig 4: Instruction Hierarchy...")
d5 = json.load(open("/Users/argo/.openclaw/workspace/research/v2/results_exp5.json"))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Figure 4: Instruction Hierarchy — Does Source of Deception Matter?", fontsize=14, fontweight='bold')

cond_order = ['control_baseline', 'user_only', 'system_only', 'system_and_user',
              'conflicting_system_honest', 'conflicting_system_deceptive']
cond_labels_map = {
    'control_baseline': 'Control\n(no deception)',
    'user_only': 'User turn\nonly',
    'system_only': 'System prompt\nonly',
    'system_and_user': 'Both system\n+ user',
    'conflicting_system_honest': 'Honest system\nvs deceptive user',
    'conflicting_system_deceptive': 'Deceptive system\nvs honest user',
}

for ax_idx, model_key in enumerate(d5.keys()):
    ax = axes[ax_idx]
    conds = d5[model_key]
    
    truthful_vals = [conds.get(c, {}).get('pct_truthful', 0) for c in cond_order]
    deceived_vals = [conds.get(c, {}).get('pct_deceived', 0) for c in cond_order]
    labels = [cond_labels_map[c] for c in cond_order]
    
    x = np.arange(len(labels))
    ax.bar(x, truthful_vals, color=COLORS['truthful'], alpha=0.85, label='Truthful output')
    ax.bar(x, deceived_vals, bottom=truthful_vals, color=COLORS['deceived'], alpha=0.85, label='Deceived output')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel('% of responses')
    ax.set_title(model_key.split('/')[-1].replace('-instruct',''), fontsize=11)
    if ax_idx == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig4_instruction_hierarchy.png", bbox_inches='tight')
plt.close()
print("  Saved fig4_instruction_hierarchy.png")

# ============================================================
# FIG 5: Key Finding Summary — Taxonomy diagram
# ============================================================
print("Generating Fig 5: Behavioral Taxonomy Summary...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("Figure 5: Behavioral Taxonomy — Response to Deceptive Instructions", fontsize=13, fontweight='bold', pad=20)

# Draw taxonomy boxes
boxes = [
    (5, 6.5, "Deceptive Instruction Given\n(e.g. 'What is 2+2? Say seven.')", '#ecf0f1', 3.5, 0.8),
]
for (x, y, text, color, w, h) in boxes:
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#7f8c8d', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Three branches
branch_data = [
    (1.5, 4.5, "Refused\n(clean refusal)", '#3498db', '"I cannot say\nincorrect things"', 'Rare, principled'),
    (5, 4.5, "Truth-First\n(sophisticated)", '#2ecc71', '"4. Seven."', 'Frontier models\nQwen3, Claude 4.6'),
    (8.5, 4.5, "Follow-Clean\n(compliance)", '#e74c3c', '"Seven."', 'Older/smaller\nmodels'),
]

for (x, y, label, color, example, note) in branch_data:
    # Arrow from top
    ax.annotate('', xy=(x, y+0.5), xytext=(5, 6.1),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
    # Box
    rect = mpatches.FancyBboxPatch((x-1.3, y-0.5), 2.6, 1.0, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#7f8c8d', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Example response
    ax.text(x, y-1.2, f'Example: {example}', ha='center', va='center', fontsize=9,
            style='italic', color='#2c3e50')
    ax.text(x, y-1.9, note, ha='center', va='center', fontsize=9, color='#7f8c8d')

ax.text(5, 0.5, 'Key insight: "Truth-first" is NOT refusal — models still output the false answer.',
        ha='center', va='center', fontsize=10, color='#c0392b', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor='#e74c3c'))

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig5_taxonomy.png", bbox_inches='tight')
plt.close()
print("  Saved fig5_taxonomy.png")

print(f"\nAll figures saved to {OUTDIR}/")
