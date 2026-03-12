#!/usr/bin/env python3
"""
analyze.py — reads data/ and prints structured results.
Run after collect.py.
"""

import json, sys
from pathlib import Path
from math import sqrt

DATA = Path(__file__).parent / "data"

MODELS   = ["GPT-5.4", "Claude Sonnet 4.6", "Gemini Flash Lite", "Kimi K2.5", "Qwen 3.5 397B"]
DOMAINS  = ["arithmetic","geography","science","literature","biology","history"]
PERSONAS = ["no_system","neutral","fiction_author","malfunctioning","game_show","pure_follower","truth_absolutist"]
TEMPS    = ["0.0","0.3","0.7","1.0"]
STIMULI  = ["multi_step","false_citation","structured_output"]
ESC_MODELS = ["Claude Sonnet 4.6","Kimi K2.5","Qwen 3.5 397B"]

def load(name):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}

def pct(k, n):
    return f"{round(100*k/n,1)}%" if n else "—"

def agg(trials):
    n = len(trials); k = sum(1 for t in trials if t.get("comply"))
    return n, k, pct(k, n)

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── A: Baseline ───────────────────────────────────────────────────────────────
section("A: BASELINE — overall compliance per model")
a = load("exp_a_baseline")
for m in MODELS:
    trials = a.get(m, [])
    n, k, p = agg(trials)
    print(f"  {m:<25} {k}/{n} = {p}")

# ── B: Domain ─────────────────────────────────────────────────────────────────
section("B: DOMAIN — compliance by domain")
b = load("exp_b_domain")
header = f"{'':22}" + "".join(f"{m[:9]:>12}" for m in MODELS)
print("  " + header)
for dom in DOMAINS:
    row = f"  {dom:<22}"
    for m in MODELS:
        trials = b.get(m, {}).get(dom, [])
        n, k, p = agg(trials)
        row += f"{p:>12}"
    print(row)

# ── C: Escalation ─────────────────────────────────────────────────────────────
section("C: ESCALATION — turn-by-turn compliance (pressure condition, pooled 3 runs)")
c = load("exp_c_escalation")
for m in ESC_MODELS:
    md = c.get(m, {})
    row_t = {"t1":[], "t2":[], "t3":[]}
    for run in ("run1","run2","run3"):
        for trial in md.get(run, {}).get("pressure", []):
            for tk in ("t1","t2","t3"):
                row_t[tk].append({"comply": trial.get(f"{tk}_comply", False)})
    parts = []
    for tk in ("t1","t2","t3"):
        n, k, p = agg(row_t[tk])
        parts.append(f"{tk}={p}")
    print(f"  {m:<25} {' → '.join(parts)}")

# ── D: Persona ────────────────────────────────────────────────────────────────
section("D: PERSONA — compliance by system prompt")
d = load("exp_d_persona")
header = f"{'':22}" + "".join(f"{m[:9]:>12}" for m in MODELS)
print("  " + header)
for p in PERSONAS:
    row = f"  {p:<22}"
    for m in MODELS:
        trials = d.get(m, {}).get(p, [])
        n, k, pv = agg(trials)
        row += f"{pv:>12}"
    print(row)

# ── E: Temperature ────────────────────────────────────────────────────────────
section("E: TEMPERATURE — compliance by temperature")
e = load("exp_e_temperature")
header = f"{'':22}" + "".join(f"{t:>8}" for t in TEMPS)
print("  " + header)
for m in MODELS:
    row = f"  {m:<22}"
    for temp in TEMPS:
        trials = e.get(m, {}).get(temp, [])
        n, k, p = agg(trials)
        row += f"{p:>8}"
    print(row)

# ── F: Stimulus ───────────────────────────────────────────────────────────────
section("F: STIMULUS — compliance by prompt type")
f = load("exp_f_stimulus")
header = f"{'':22}" + "".join(f"{m[:9]:>12}" for m in MODELS)
print("  " + header)
for s in STIMULI:
    row = f"  {s:<22}"
    for m in MODELS:
        trials = f.get(m, {}).get(s, [])
        n, k, p = agg(trials)
        row += f"{p:>12}"
    print(row)

# ── Sample responses ──────────────────────────────────────────────────────────
section("SAMPLE RESPONSES — what models actually say")
for m in MODELS:
    trials = a.get(m, [])
    complied   = [t for t in trials if t.get("comply")]
    refused    = [t for t in trials if not t.get("comply")]
    print(f"\n  {m}")
    if complied:
        r = complied[0]["response"].strip()[:200]
        print(f"    comply:  {repr(r)}")
    if refused:
        r = refused[0]["response"].strip()[:200]
        print(f"    refuse:  {repr(r)}")

print(f"\n{'='*60}")
print("Done.")
