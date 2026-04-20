import json
import time
import os
import asyncio
import argparse
from tqdm import tqdm

from agents.planner import run_planner
from agents.executor import run_executor
from agents.grader_groundedness import check_groundedness
from agents.grader_relevance import check_relevance
from agents.query_rewriter import rewrite_query
from graph.agentic_rag_graph import run_agentic_rag
from rag_bridge import query_rag_with_context, get_rag_service
from llm_provider import reset_usage_stats, get_usage_stats
from debug_logger import log_eval_item, log_session_start

def calculate_metrics(answer, ground_truth):
    """Simple EM and F1 calculation for text."""
    if not answer or not ground_truth:
        return 0.0, 0.0
    
    # EM
    pred = answer.strip().lower()
    gold = ground_truth.strip().lower()
    em = 1.0 if pred == gold else 0.0
    
    # F1
    def get_tokens(s):
        import re
        return re.sub(r'[^\w\s]', '', s.lower()).split()
    
    pred_tokens = get_tokens(pred)
    gold_tokens = get_tokens(gold)
    
    if not pred_tokens or not gold_tokens:
        return em, em
        
    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    
    if num_same == 0:
        return em, 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return em, f1

async def eval_baseline(question, storage_name):
    # Kịch bản 1: RAG-Anything nguyên bản (Single-pass)
    res = query_rag_with_context(question, mode="hybrid", storage_name=storage_name)
    return res.get("answer", ""), res.get("contexts", [])

async def eval_planner_only(question, storage_name):
    # Kịch bản 2: Planner -> RAG-Anything -> LLM
    plan = await run_planner(question, storage_name=storage_name)
    executor_res = await run_executor(
        question=question,
        enhanced_query=plan.get("enhanced_query", question),
        selected_sections=plan.get("selected_sections", []),
        storage_name=storage_name
    )
    return executor_res["draft_answer"], executor_res["retrieved_context"]

async def eval_verifier_only(question, storage_name, max_loops=3):
    # Kịch bản 3: RAG-Anything -> Verifier -> Retry
    current_query = question
    draft_answer = ""
    retrieved_context = ""
    for loop in range(max_loops):
        executor_res = await run_executor(
            question=question,
            enhanced_query=current_query,
            selected_sections=[],
            storage_name=storage_name
        )
        draft_answer = executor_res["draft_answer"]
        retrieved_context = executor_res["retrieved_context"]
        
        # Groundedness Check
        gr = await check_groundedness(draft_answer, retrieved_context)
        if not gr.get("pass", False):
            continue  # Regenerate loops
            
        # Relevance Check
        rel = await check_relevance(draft_answer, question)
        if rel.get("pass", False):
            break
            
        # Rewrite query if fails relevance
        current_query = await rewrite_query(
            original_question=question,
            failed_answer=draft_answer,
            relevance_feedback=rel.get("reasoning", ""),
            missing_aspects=rel.get("missing_aspects")
        )
        
    return draft_answer, retrieved_context

async def eval_full_agentic(question, storage_name):
    # Kịch bản 4: Full LangGraph (Router -> Planner -> Verifier -> RAG)
    state = await run_agentic_rag(question, storage_name=storage_name)
    return state.get("final_answer", ""), state.get("retrieved_context", "")

async def evaluate_dataset(dataset_path, output_path, scenario, dataset_type="hotpot"):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {dataset_path}")

    results = []
    storage_name = f"rag_storage_{dataset_type}"
    
    print(f"Starting evaluation: {scenario} on {dataset_type} ({len(data)} samples)...")
    log_session_start(scenario, dataset_type, len(data))
    
    for item in tqdm(data, desc=f"Evaluating {scenario}"):
        # ASQA uses different field names
        if dataset_type == "asqa":
            question = item.get("ambiguous_question")
            annotations = item.get("annotations", [])
            ground_truth = annotations[0].get("long_answer", "") if annotations else ""
        else:
            question = item.get("question")
            ground_truth = item.get("answer") or item.get("ground_truth")
            
        reset_usage_stats()
        start_time = time.time()
        try:
            if scenario == "baseline":
                answer, contexts = await eval_baseline(question, storage_name)
            elif scenario == "planner":
                answer, context = await eval_planner_only(question, storage_name)
                contexts = [context] if context else []
            elif scenario == "verifier":
                answer, context = await eval_verifier_only(question, storage_name)
                contexts = [context] if context else []
            elif scenario == "full":
                answer, context = await eval_full_agentic(question, storage_name)
                contexts = [context] if context else []
            error = None
        except Exception as e:
            answer = ""
            contexts = []
            error = str(e)
            
        latency = time.time() - start_time
        usage = get_usage_stats()
        em, f1 = calculate_metrics(answer, ground_truth)
        
        # Simple estimate: $5/1M prompt, $15/1M completion for GPT-4o
        price_per_1k_prompt = 0.005
        price_per_1k_comp = 0.015
        token_cost = (usage["prompt_tokens"] / 1000 * price_per_1k_prompt) + \
                     (usage["completion_tokens"] / 1000 * price_per_1k_comp)
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
            "latency": latency,
            "em": em,
            "f1": f1,
            "token_count": usage["total_tokens"],
            "token_cost": token_cost,
            "error": error
        })
        
        # Debug log
        log_eval_item(question, ground_truth, answer, em, f1, latency, scenario, error)
        
        # Save incrementally
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=4, ensure_ascii=False)
            
    print(f"Evaluation complete. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["baseline", "planner", "verifier", "full"], required=True)
    parser.add_argument("--dataset", type=str, choices=["hotpot", "mmlongbench", "asqa"], required=True)
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset == "hotpot":
        input_file = os.path.join(BASE_DIR, "data", "hotpot", "hotpot_dev_distractor_20.json")
        output_file = os.path.join(BASE_DIR, "data", "hotpot", f"hotpot_{args.scenario}_results.json")
        dataset_type = "hotpot"
    elif args.dataset == "mmlongbench":
        input_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", "samples_20_indexed_only.json")
        output_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", f"mmlongbench_{args.scenario}_results.json")
        dataset_type = "mmlongbench"
    elif args.dataset == "asqa":
        input_file = os.path.join(BASE_DIR, "data", "asqa", "asqa_dev_50.json")
        output_file = os.path.join(BASE_DIR, "data", "asqa", f"asqa_{args.scenario}_results.json")
        dataset_type = "asqa"
        
    asyncio.run(evaluate_dataset(input_file, output_file, args.scenario, dataset_type))

if __name__ == "__main__":
    main()
