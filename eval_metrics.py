import json
import os
import argparse
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import string
import collections
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to json results from evaluation.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file.")
    args = parser.parse_args()

    with open(args.results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare lists for dataset
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []

    em_scores = []
    f1_scores = []

    for item in data:
        q = item.get("question", "")
        a = item.get("answer", "")
        c = item.get("contexts", [""])
        if not c: c = [""]
        gt = item.get("ground_truth", "")
        lat = item.get("latency", 0.0)

        questions.append(q)
        answers.append(a)
        contexts.append(c)
        ground_truths.append([gt])
        latencies.append(lat)

        em_scores.append(compute_exact(gt, a))
        f1_scores.append(compute_f1(gt, a))

    # Initialize OpenAI models for Ragas via Langchain
    # Requires OPENAI_API_KEY to be set
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(eval_data)

    print("Running Ragas evaluation...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevance],
        llm=llm,
        embeddings=embeddings
    )

    df_out = result.to_pandas()
    
    # Append custom fields
    df_out["EM"] = em_scores
    df_out["F1"] = f1_scores
    df_out["Latency"] = latencies
    
    df_out.to_csv(args.output_csv, index=False)
    print(f"Metrics saved to {args.output_csv}")

    # Summary
    print("----- SUMMARY -----")
    print(f"Average EM: {sum(em_scores)/len(em_scores):.4f}")
    print(f"Average F1: {sum(f1_scores)/len(f1_scores):.4f}")
    print(f"Average Latency: {sum(latencies)/len(latencies):.2f}s")
    for metric_name, metric_val in result.items():
        print(f"{metric_name}: {metric_val:.4f}")

if __name__ == "__main__":
    main()
