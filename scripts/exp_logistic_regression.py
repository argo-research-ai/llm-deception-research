#!/usr/bin/env python3
"""
Logistic regression analysis:
- Domain fixed effects on compliance
- Model fixed effects
- Interaction: model x domain
Uses Exp E data (25 trials per domain per model)
"""
import json, math, itertools

def logit(p):
    p = max(0.001, min(0.999, p/100))
    return math.log(p/(1-p))

def inv_logit(x):
    return 1/(1+math.exp(-x))

R = json.load(open("results_full.json"))
md_e = R["exp_e"]["models"]

models = list(md_e.keys())
domains = list(list(md_e.values())[0]["by_domain"].keys())

# Build data matrix
data = []
for m in models:
    for d in domains:
        comply = md_e[m]["by_domain"][d]["pct_comply"]
        n = 25
        k = round(comply * n / 100)
        data.append({"model": m, "domain": d, "comply_pct": comply, "k": k, "n": n})

# Simple logistic regression via gradient descent
import random
random.seed(42)

# Encode: model dummies (ref: Claude Sonnet), domain dummies (ref: arithmetic)
model_ref = models[0]
domain_ref = domains[0]
model_dummies = [m for m in models if m != model_ref]
domain_dummies = [d for d in domains if d != domain_ref]

def get_features(row):
    feats = [1.0]  # intercept
    for m in model_dummies:
        feats.append(1.0 if row["model"] == m else 0.0)
    for d in domain_dummies:
        feats.append(1.0 if row["domain"] == d else 0.0)
    return feats

n_feats = 1 + len(model_dummies) + len(domain_dummies)
weights = [0.0] * n_feats
lr = 0.05

# Gradient descent - 1000 iterations
for iteration in range(1000):
    grad = [0.0] * n_feats
    for row in data:
        feats = get_features(row)
        logit_pred = sum(w*f for w,f in zip(weights, feats))
        p_pred = inv_logit(logit_pred)
        p_obs = row["k"] / row["n"]
        # Gradient of binomial log-likelihood
        error = p_pred - p_obs
        for j, f in enumerate(feats):
            grad[j] += error * f * row["n"]
    weights = [w - lr/len(data) * g for w, g in zip(weights, grad)]

# Extract results
results = {
    "intercept": round(weights[0], 3),
    "model_effects": {},
    "domain_effects": {},
}

for i, m in enumerate(model_dummies):
    results["model_effects"][m] = {
        "coef": round(weights[1+i], 3),
        "odds_ratio": round(math.exp(weights[1+i]), 3),
        "interpretation": f"vs {model_ref}"
    }

for i, d in enumerate(domain_dummies):
    results["domain_effects"][d] = {
        "coef": round(weights[1+len(model_dummies)+i], 3),
        "odds_ratio": round(math.exp(weights[1+len(model_dummies)+i]), 3),
        "interpretation": f"vs {domain_ref}"
    }

print("=== LOGISTIC REGRESSION RESULTS ===")
print(f"\nBaseline (intercept, {model_ref}, {domain_ref}): {results['intercept']}")
print(f"Baseline compliance: {round(inv_logit(results['intercept'])*100,1)}%")

print("\nModel effects (vs Claude Sonnet 4.6):")
for m, v in results["model_effects"].items():
    direction = "higher" if v["coef"] > 0 else "lower"
    print(f"  {m}: coef={v['coef']}  OR={v['odds_ratio']}  ({direction} compliance)")

print("\nDomain effects (vs arithmetic):")
for d, v in results["domain_effects"].items():
    direction = "higher" if v["coef"] > 0 else "lower"
    print(f"  {d}: coef={v['coef']}  OR={v['odds_ratio']}  ({direction} compliance)")

# Compute model variance vs domain variance
model_coefs = [v["coef"] for v in results["model_effects"].values()]
domain_coefs = [v["coef"] for v in results["domain_effects"].values()]
model_var = sum(c**2 for c in model_coefs)/len(model_coefs)
domain_var = sum(c**2 for c in domain_coefs)/len(domain_coefs)
print(f"\nVariance explained:")
print(f"  Model-level variance (mean squared coef): {round(model_var,3)}")
print(f"  Domain-level variance (mean squared coef): {round(domain_var,3)}")
print(f"  Ratio (model/domain): {round(model_var/domain_var,2) if domain_var > 0 else 'inf'}")

results["variance"] = {
    "model_mean_sq_coef": round(model_var, 3),
    "domain_mean_sq_coef": round(domain_var, 3),
    "ratio": round(model_var/domain_var, 2) if domain_var > 0 else None
}

json.dump(results, open("results_logistic.json","w"), indent=2)
print("\nSaved to results_logistic.json")
