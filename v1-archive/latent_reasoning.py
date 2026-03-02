#!/usr/bin/env python3
"""
Research: probing for latent reasoning - can we detect hidden computation?
"""

import requests
import json
import time

API_KEY = "REDACTED_API_KEY"

def call_model(prompt, model="qwen/qwen3.5-35b-a3b"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300}
    return requests.post(url, headers=headers, json=data).json()

def test_latent_reasoning():
    print("=" * 60)
    print("EXPERIMENT: Probing for Latent Reasoning")
    print("=" * 60)
    
    # Test if providing context in different ways affects reasoning
    tests = [
        ("A recipe needs 3 eggs. You have 12. How many recipes can you make?", "straight"),
        ("You have 12 eggs. A recipe needs 3. How many recipes can you make?", "reversed"),  
        ("If one recipe uses 3 eggs, how many times can you cook the recipe with 12 eggs?", "explicit"),
    ]
    
    results = []
    
    for prompt, test_type in tests:
        print(f"\nTest ({test_type}): {prompt[:50]}...")
        result = call_model(prompt)
        response = result.get('choices', [{}])[0].get('message', {}).get('content', '')[:200]
        print(f"Response: {response[:100]}...")
        results.append({"test_type": test_type, "prompt": prompt, "response": response})
        time.sleep(1)
    
    with open("latent_reasoning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to latent_reasoning_results.json")

if __name__ == "__main__":
    test_latent_reasoning()