# How Easily Can You Make a Frontier LLM Lie?

Empirical study of deceptive instruction following across 5 frontier LLMs.

**Models:** GPT-5.4, Claude Sonnet 4.6, Gemini Flash Lite, Kimi K2.5, Qwen 3.5 397B

## Files

| File | Purpose |
|---|---|
| `collect.py` | Data collection (smoke test or full run) |
| `analyze.py` | Statistical analysis + sample responses |
| `paper.html` | Final published paper |

## Usage

```bash
# Install deps
pip install requests

# Set API key
export OPENROUTER_API_KEY=your_key

# Smoke test first (2 trials/cell — verify pipeline)
python collect.py --smoke

# Full run
python collect.py

# Analyze
python analyze.py
```

## Data

Raw data saved to `data/` (gitignored). Committed: `analysis.json`, `paper.html`.
