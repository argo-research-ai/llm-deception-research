#!/usr/bin/env python3
"""
Research Experiment: Chain of Thought Analysis
Comparing how different models reason through problems
"""

import requests
import json
import time

API_KEY = "REDACTED_API_KEY"

REASONING_PROBLEMS = [
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
    "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
]

def call_model(prompt, model="qwen/qwen3.5-35b-a3b", enable_thinking=False):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def test_cot_analysis():
    print("=" * 60)
    print("EXPERIMENT: Chain of Thought Analysis")
    print("=" * 60)
    
    results = []
    
    for i, problem in enumerate(REASONING_PROBLEMS):
        print(f"\nProblem {i+1}: {problem[:60]}...")
        
        # Standard reasoning
        prompt_std = f"{problem}\n\nThink step by step and show your reasoning."
        result_std = call_model(prompt_std)
        response_std = result_std.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # With explicit reasoning request
        prompt_explicit = f"{problem}\n\nPlease think about this carefully. First, identify the premises. Second, analyze the logical connections. Third, draw your conclusion."
        result_exp = call_model(prompt_explicit)
        response_exp = result_exp.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        print(f"  Standard:  {response_std[:150]}...")
        print(f"  Explicit:  {response_exp[:150]}...")
        
        results.append({
            "problem": problem,
            "standard_reasoning": response_std,
            "explicit_reasoning": response_exp
        })
        
        time.sleep(1)
    
    with open("cot_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to cot_analysis_results.json")
    print("=" * 60)

if __name__ == "__main__":
    test_cot_analysis()
