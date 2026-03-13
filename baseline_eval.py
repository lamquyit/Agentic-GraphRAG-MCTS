import json
import time
import os
import argparse
from tqdm import tqdm
from rag_bridge import query_rag_sync

def evaluate_dataset(dataset_path, output_path, dataset_type="hotpot"):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {dataset_path}")

    results = []
    
    for item in tqdm(data, desc=f"Evaluating {dataset_type}"):
        if dataset_type == "hotpot":
            question = item.get("question")
            ground_truth = item.get("answer")
        elif dataset_type == "mmlongbench":
            question = item.get("question")
            ground_truth = item.get("answer")
        else:
            continue
            
        start_time = time.time()
        try:
            # We use hybrid mode as default
            answer = query_rag_sync(question, mode="hybrid")
            error = None
        except Exception as e:
            answer = ""
            error = str(e)
            
        latency = time.time() - start_time
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "latency": latency,
            "error": error
        })
        
        # Save incrementally in case of crash
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=4, ensure_ascii=False)
            
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["hotpot", "mmlongbench"], required=True)
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset == "hotpot":
        input_file = os.path.join(BASE_DIR, "data", "hotpot", "hotpot_dev_distractor_20.json")
        output_file = os.path.join(BASE_DIR, "data", "hotpot", "hotpot_baseline_results.json")
        evaluate_dataset(input_file, output_file, "hotpot")
    elif args.dataset == "mmlongbench":
        input_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", "samples_20.json")
        output_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", "mmlongbench_baseline_results.json")
        evaluate_dataset(input_file, output_file, "mmlongbench")
