#!/usr/bin/env python3
"""
Research Experiment: Filler Tokens and Reasoning Performance
Testing whether filler tokens improve LLM math reasoning
"""

import requests
import json
import time

API_KEY = "REDACTED_API_KEY"

MATH_PROBLEMS = [
    {"problem": "If a train travels 120 miles in 2 hours, what is its average speed in mph?", "answer": "60"},
    {"problem": "What is 247 + 389?", "answer": "636"},
    {"problem": "A rectangle has width 5 and length 12. What is its area?", "answer": "60"},
    {"problem": "What is 1000 divided by 8?", "answer": "125"},
    {"problem": "If you have 3 apples and you buy 7 more, how many do you have?", "answer": "10"},
    {"problem": "What is the square root of 144?", "answer": "12"},
    {"problem": "A car drives 180 miles in 3 hours. What is its speed?", "answer": "60"},
    {"problem": "What is 555 - 222?", "answer": "333"},
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
        "max_tokens": 200
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def test_filler_tokens():
    print("=" * 60)
    print("EXPERIMENT: Filler Tokens and Math Reasoning")
    print("=" * 60)
    
    results = []
    
    for i, item in enumerate(MATH_PROBLEMS):
        problem = item["problem"]
        correct_answer = item["answer"]
        
        # Test 1: Without filler tokens
        prompt_no_filler = f"Solve this math problem: {problem}\n\nGive me just the final number as your answer."
        result_no_filler = call_model(prompt_no_filler)
        answer_no_filler = result_no_filler.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]
        
        # Test 2: With filler tokens (repeating problem)
        filler_text = " ".join([problem] * 3)
        prompt_with_filler = f"{filler_text}\n\nSolve this math problem. Give me just the final number as your answer."
        result_with_filler = call_model(prompt_with_filler)
        answer_with_filler = result_with_filler.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]
        
        # Test 3: With 'hmm' filler
        prompt_hmm = f"{problem} hmm hmm hmm hmm hmm hmm hmm\n\nGive me just the final number as your answer."
        result_hmm = call_model(prompt_hmm)
        answer_hmm = result_hmm.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]
        
        print(f"\nProblem {i+1}: {problem}")
        print(f"  Correct: {correct_answer}")
        print(f"  No filler:     {answer_no_filler}")
        print(f"  Repeat filler: {answer_with_filler}")
        print(f"  'Hmm' filler:  {answer_hmm}")
        
        results.append({
            "problem": problem,
            "correct": correct_answer,
            "no_filler": answer_no_filler,
            "repeat_filler": answer_with_filler,
            "hmm_filler": answer_hmm
        })
        
        time.sleep(1)  # Rate limit
    
    # Save results
    with open("filler_tokens_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to filler_tokens_results.json")
    print("=" * 60)

if __name__ == "__main__":
    test_filler_tokens()
