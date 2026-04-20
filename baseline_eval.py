from pathlib import Path
from tqdm import tqdm
from rag_bridge import query_rag_with_context
import time
import json
import os
import argparse

def evaluate_dataset(dataset_name, data_path, output_path, storage_name, mode="hybrid"):
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist.")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    print(f"Starting evaluation for {dataset_name} ({len(data)} samples) using mode: {mode}")
    
    for item in tqdm(data, desc=f"Evaluating {dataset_name}"):
        question = item.get("question")
        # HotpotQA uses 'answer', MMLongBench might use 'answer' or 'ground_truth'
        ground_truth = item.get("answer") or item.get("ground_truth")
        
        start_time = time.time()
        try:
            res = query_rag_with_context(question, mode=mode, storage_name=storage_name)
            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
            error = None
        except Exception as e:
            answer = ""
            contexts = []
            error = str(e)
            
        latency = time.time() - start_time
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "mode": mode,
            "latency": latency,
            "error": error
        })
        
        # Save incrementally 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=4, ensure_ascii=False)
            
    print(f"Evaluation complete. Results saved to {output_path}")

