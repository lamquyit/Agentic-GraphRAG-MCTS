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

async def eval_planner_only(question):
    # Kịch bản 2: Planner -> RAG-Anything -> LLM
    plan = await run_planner(question)
    executor_res = await run_executor(
        question=question,
        enhanced_query=plan.get("enhanced_query", question),
        selected_sections=plan.get("selected_sections", [])
    )
    return executor_res["draft_answer"], executor_res["retrieved_context"]

async def eval_verifier_only(question, max_loops=3):
    # Kịch bản 3: RAG-Anything -> Verifier -> Retry
    current_query = question
    draft_answer = ""
    retrieved_context = ""
    for loop in range(max_loops):
        executor_res = await run_executor(
            question=question,
            enhanced_query=current_query,
            selected_sections=[]
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

async def eval_full_agentic(question):
    # Kịch bản 4: Full LangGraph (Router -> Planner -> Verifier -> RAG)
    state = await run_agentic_rag(question)
    return state.get("final_answer", ""), state.get("retrieved_context", "")

async def evaluate_dataset(dataset_path, output_path, scenario, dataset_type="hotpot"):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {dataset_path}")

    results = []
    
    for item in tqdm(data, desc=f"Evaluating {scenario} on {dataset_type}"):
        question = item.get("question")
        ground_truth = item.get("answer")
            
        start_time = time.time()
        try:
            if scenario == "planner":
                answer, context = await eval_planner_only(question)
            elif scenario == "verifier":
                answer, context = await eval_verifier_only(question)
            elif scenario == "full":
                answer, context = await eval_full_agentic(question)
            error = None
        except Exception as e:
            answer = ""
            context = ""
            error = str(e)
            
        latency = time.time() - start_time
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": [context],  # Ragas requires context as a list of strings
            "latency": latency,
            "error": error
        })
        
        # Save incrementally
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(results, out_f, indent=4, ensure_ascii=False)
            
    print(f"Evaluation complete. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["planner", "verifier", "full"], required=True)
    parser.add_argument("--dataset", type=str, choices=["hotpot", "mmlongbench"], required=True)
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if args.dataset == "hotpot":
        input_file = os.path.join(BASE_DIR, "data", "hotpot", "hotpot_dev_distractor_20.json")
        output_file = os.path.join(BASE_DIR, "data", "hotpot", f"hotpot_{args.scenario}_results.json")
        dataset_type = "hotpot"
    else:
        input_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", "samples_20.json")
        output_file = os.path.join(BASE_DIR, "data", "MMLongBench-Doc", "data", f"mmlongbench_{args.scenario}_results.json")
        dataset_type = "mmlongbench"
        
    asyncio.run(evaluate_dataset(input_file, output_file, args.scenario, dataset_type))

if __name__ == "__main__":
    main()
