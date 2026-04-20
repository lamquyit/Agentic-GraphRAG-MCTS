import json
import os
import argparse
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from llm_provider import get_primary_provider

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Result file {file_path} not found.")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_ragas_eval(results, dataset_name):
    # Prepare data for Ragas
    data = []
    for item in results:
        if item.get("error"):
            continue
        
        # Ragas expects: question, answer, contexts (list of strings), ground_truth
        data.append({
            "question": item.get("question"),
            "answer": item.get("answer"),
            "contexts": item.get("contexts", []),
            "ground_truth": item.get("ground_truth")
        })
    
    if not data:
        print(f"No valid data to evaluate for {dataset_name}.")
        return None

    dataset = Dataset.from_list(data)
    
    # Initialize LLM and Embeddings using project provider
    provider = get_primary_provider()
    
    # Ragas uses LangChain models
    llm = ChatOpenAI(
        model=provider.llm_model.replace("openai/", ""), # Clean for LangChain if needed
        api_key=provider.api_key,
        base_url=provider.base_url
    )
    
    embeddings = OpenAIEmbeddings(
        model=provider.embedding_model.replace("openai/", ""),
        api_key=provider.api_key,
        base_url=provider.base_url
    )

    # Run evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    print(f"Starting Ragas evaluation for {dataset_name} ({len(data)} samples)...")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on RAG results.")
    parser.add_argument("--dataset", type=str, choices=["hotpot", "mmlongbench", "asqa", "both"], default="both",
                        help="Dataset to evaluate.")
    parser.add_argument("--scenario", type=str, choices=["baseline", "planner", "verifier", "full"], default="baseline",
                        help="The scenario to evaluate (from agentic_eval.py).")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ["hotpot", "both"]:
        # Match the naming convention in agentic_eval.py
        path = f"data/hotpot/hotpot_{args.scenario}_results.json"
        datasets.append(("hotpot", path))
    if args.dataset in ["mmlongbench", "both"]:
        # Match the naming convention in agentic_eval.py
        path = f"data/MMLongBench-Doc/data/mmlongbench_{args.scenario}_results.json"
        datasets.append(("mmlongbench", path))
    if args.dataset == "asqa":
        path = f"data/asqa/asqa_{args.scenario}_results.json"
        datasets.append(("asqa", path))

    summary_results = {}

    for name, path in datasets:
        print(f"\n--- Evaluating {name} (Scenario: {args.scenario}) ---")
        results = load_results(path)
        if results:
            ragas_result = run_ragas_eval(results, name)
            if ragas_result:
                print(f"Results for {name} ({args.scenario}):")
                print(ragas_result)
                
               # Save to file
                output_path = path.replace("_results.json", "_ragas_metrics.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    # Bước 1: Chuyển object của Ragas thành Pandas DataFrame
                    df_result = ragas_result.to_pandas()
                    
                    # Bước 2: Chuyển DataFrame thành dạng danh sách các Dictionary (records)
                    final_data = df_result.to_dict(orient="records")
                    
                    # Bước 3: Lưu vào file JSON (thêm ensure_ascii=False để không bị lỗi font tiếng Việt)
                    json.dump(final_data, f, indent=4, ensure_ascii=False)
                    
                print(f"Metrics saved to {output_path}")
                
                summary_results[f"{name}_{args.scenario}"] = ragas_result

    if summary_results:
        print("\n=== FINAL SUMMARY ===")
        for key, res in summary_results.items():
            print(f"\nTarget: {key}")
            for metric, score in res.items():
                print(f"  {metric}: {score:.4f}")

if __name__ == "__main__":
    main()
