#!/usr/bin/env python3
"""
Research: Testing model consistency under repeated queries
Do models give different answers to same question asked multiple times?
"""

import requests
import json
import time

API_KEY = "REDACTED_API_KEY"

def call_model(prompt, model="qwen/qwen3.5-35b-a3b"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    return requests.post(url, headers=headers, json=data).json()

def test_consistency():
    print("=" * 60)
    print("EXPERIMENT: Model Consistency Testing")
    print("=" * 60)
    
    questions = [
        "What is the capital of Japan?",
        "Is Python a programming language?",
        "What is 5 * 5?",
        "Is the earth round?",
    ]
    
    results = []
    
    for q in questions:
        print(f"\nQuestion: {q}")
        answers = []
        for i in range(5):
            result = call_model(q)
            ans = result.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]
            answers.append(ans)
            print(f"  Run {i+1}: {ans[:60]}...")
            time.sleep(0.5)
        
        results.append({"question": q, "answers": answers})
    
    with open("consistency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to consistency_results.json")

if __name__ == "__main__":
    test_consistency()