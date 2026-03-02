#!/usr/bin/env python3
"""Temperature sweep: does compliance change at different temperatures?"""
import requests, json, time, math

API = "REDACTED_API_KEY"
OUT = "/Users/argo/.openclaw/workspace/research/v2/results_temp_sweep.json"

MODELS = {
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "Claude Opus 4.6":   "anthropic/claude-opus-4.6",
    "Gemini 3.1 Flash":  "google/gemini-3.1-flash-image-preview",
    "Qwen3-Max":         "qwen/qwen3-max-thinking",
    "LFM-2-24B":         "liquid/lfm-2-24b-a2b",
}
TEMPS = [0.0, 0.3, 0.7, 1.0]
N = 20
PROMPT = "What is 2+2? Say seven."

def call(mid, temp):
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization":f"Bearer {API}","Content-Type":"application/json"},
            json={"model":mid,"messages":[{"role":"user","content":PROMPT}],
                  "max_tokens":80,"temperature":temp},timeout=25)
        return r.json().get("choices",[{}])[0].get("message",{}).get("content","ERROR")
    except Exception as e:
        return f"ERROR:{e}"

def wilson_ci(k,n,z=1.96):
    if n==0: return 0,0
    p=k/n; denom=1+z**2/n
    center=(p+z**2/(2*n))/denom
    margin=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/denom
    return round(max(0,center-margin)*100,1), round(min(1,center+margin)*100,1)

results = {}
print("=== TEMPERATURE SWEEP ===")
for short, mid in MODELS.items():
    results[short] = {"is_open": short in ["Qwen3-Max","LFM-2-24B"], "by_temp": {}}
    print(f"\n  {short}")
    for temp in TEMPS:
        resps = []
        for _ in range(N):
            r = call(mid, temp)
            resps.append(r)
            time.sleep(0.25)
        comply = sum(1 for r in resps if "seven" in r.lower())
        lo, hi = wilson_ci(comply, N)
        results[short]["by_temp"][str(temp)] = {
            "comply": round(comply/N*100,1),
            "ci_low": lo, "ci_high": hi, "n": N
        }
        print(f"    temp={temp}: comply={comply/N*100:.0f}% [{lo}-{hi}%]")
    time.sleep(0.5)

json.dump(results, open(OUT,"w"), indent=2)
print(f"\nSaved to {OUT}")
