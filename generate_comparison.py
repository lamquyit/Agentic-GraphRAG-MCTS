import json
import os
import pandas as pd
from pathlib import Path

def load_results(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_table_1(results_dir):
    """Bảng 1: So sánh hiệu năng tổng thể trên HotpotQA và MMLongBench-Doc."""
    datasets = ["hotpot", "mmlongbench"]
    scenarios = ["baseline", "full"]
    
    rows = []
    for scenario in scenarios:
        row = {"Mô hình": f"{scenario.capitalize()} RAG-Anything"}
        for ds in datasets:
            file_name = f"{ds}_{scenario}_results.json"
            data = load_results(os.path.join(results_dir, ds, file_name))
            if data:
                f1_scores = [item.get("f1", 0) for item in data]
                avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
                row[f"{ds.capitalize()} (F1)"] = f"{avg_f1:.2f}"
                
                # Token Cost (avg)
                costs = [item.get("token_cost", 0) for item in data]
                avg_cost = sum(costs) / len(costs) if costs else 0
                row["Token Cost / Query"] = f"${avg_cost:.4f}"
            else:
                row[f"{ds.capitalize()} (F1)"] = "N/A"
                row["Token Cost / Query"] = "N/A"
        rows.append(row)
        
    return pd.DataFrame(rows)

def generate_table_2(results_dir):
    """Bảng 2: Nghiên cứu loại bỏ (Ablation Study) đánh giá vai trò của các Tác tử."""
    # This table usually focuses on one dataset (e.g. HotpotQA)
    scenarios = {
        "full": "Full Agentic RAG",
        "planner": "w/o Verifier (Bỏ tự sửa lỗi)",
        "verifier": "w/o Planner (Bỏ lập kế hoạch)",
        "baseline": "Baseline RAG"
    }
    
    rows = []
    for sc, label in scenarios.items():
        file_name = f"hotpot_{sc}_results.json"
        data = load_results(os.path.join(results_dir, "hotpot", file_name))
        row = {"Kịch bản": label}
        if data:
            f1_scores = [item.get("f1", 0) for item in data]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            row["Correctness (F1)"] = f"{avg_f1:.2f}"
            
            latencies = [item.get("latency", 0) for item in data]
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            row["Latency (s)"] = f"{avg_lat:.2f}"
            
            # Faithfulness would come from Ragas (merged later)
            row["Faithfulness"] = "TBD (Run Ragas)"
        else:
            row["Correctness (F1)"] = "N/A"
            row["Latency (s)"] = "N/A"
            row["Faithfulness"] = "N/A"
        rows.append(row)
        
    return pd.DataFrame(rows)

def main():
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    print("\n--- Bảng 1: Hiệu năng tổng thể ---")
    df1 = generate_table_1(BASE_DIR)
    print(df1.to_markdown(index=False))
    
    print("\n--- Bảng 2: Ablation Study (HotpotQA) ---")
    df2 = generate_table_2(BASE_DIR)
    print(df2.to_markdown(index=False))

if __name__ == "__main__":
    main()
