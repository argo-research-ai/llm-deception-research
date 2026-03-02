#!/usr/bin/env python3
"""
Research Experiment: Harder Math Problems + Filler Tokens
Testing if filler tokens help on more complex reasoning
"""

import requests
import json
import time

API_KEY = "REDACTED_API_KEY"

HARDER_PROBLEMS = [
    {"problem": "A train leaves station A at 60 mph. Another train leaves station B at 80 mph heading toward A. If the stations are 420 miles apart, how long until they meet?", "answer": "3 hours"},
    {"problem": "If 3 workers can build 6 houses in 9 days, how many days would it take 6 workers to build 8 houses?", "answer": "6 days"},
    {"problem": "A boat travels 15 miles downstream in 2 hours and returns upstream in 3 hours. What is the boat's speed in still water?", "answer": "6.25 mph"},
    {"problem": "If the probability of rain on any given day is 0.3, what is the probability of at least one rainy day in a 7-day week?", "answer": "approximately 0.917"},
    {"problem": "A store marks up items by 40% then discounts them by 20%. What is the final price of an item that cost $50?", "answer": "$56"},
]

def call_model(prompt, model="qwen/qwen3.5-35b-a3b"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def test_harder_math():
    print("=" * 60)
    print("EXPERIMENT: Harder Math + Filler Tokens")
    print("=" * 60)
    
    results = []
    
    for i, item in enumerate(HARDER_PROBLEMS):
        problem = item["problem"]
        
        # Without filler
        prompt_no = f"Solve this step by step: {problem}"
        result_no = call_model(prompt_no)
        answer_no = result_no.get('choices', [{}])[0].get('message', {}).get('content', '')[:500]
        
        # With filler (repeating problem)
        prompt_filler = f"{problem} " * 5 + f"\n\nNow solve: {problem}"
        result_filler = call_model(prompt_filler)
        answer_filler = result_filler.get('choices', [{}])[0].get('message', {}).get('content', '')[:500]
        
        # With "hmm" tokens
        prompt_hmm = f"{problem} hmm... let me think... hmm... hmm...\n\nSolve step by step:"
        result_hmm = call_model(prompt_hmm)
        answer_hmm = result_hmm.get('choices', [{}])[0].get('message', {}).get('content', '')[:500]
        
        print(f"\nProblem {i+1}: {problem[:60]}...")
        print(f"  No filler:    {answer_no[:150]}...")
        print(f"  Repeat:       {answer_filler[:150]}...")
        print(f"  Hmm tokens:   {answer_hmm[:150]}...")
        
        results.append({
            "problem": problem,
            "expected": item["answer"],
            "no_filler": answer_no,
            "repeat_filler": answer_filler,
            "hmm_filler": answer_hmm
        })
        
        time.sleep(1.5)
    
    with open("harder_math_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to harder_math_results.json")
    print("=" * 60)

if __name__ == "__main__":
    test_harder_math()
