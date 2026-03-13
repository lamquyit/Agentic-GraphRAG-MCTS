import json
import random
import os

random.seed(42)  # For reproducible sampling

def sample_json_list(input_path, output_path, fraction=0.2):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {input_path}")
    
    sample_size = int(len(data) * fraction)
    sampled_data = random.sample(data, sample_size)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(sampled_data)} items to {output_path}")

def sample_hotpot_dict(input_path, output_path, fraction=0.2):
    # HotpotQA data might be a list directly, let's verify
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"Loaded {len(data)} items from {input_path}")
        sample_size = int(len(data) * fraction)
        sampled_data = random.sample(data, sample_size)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(sampled_data)} items to {output_path}")
    else:
        print(f"Warning: {input_path} is not a list. Type is {type(data)}")

if __name__ == "__main__":
    RAG_UPGRADE_DIR = "/home/lamquy/Project/RAG/RAG_upgrade"
    
    mmlongbench_path = os.path.join(RAG_UPGRADE_DIR, "data", "MMLongBench-Doc", "data", "samples.json")
    mmlongbench_20_path = os.path.join(RAG_UPGRADE_DIR, "data", "MMLongBench-Doc", "data", "samples_20.json")

    hotpot_path = os.path.join(RAG_UPGRADE_DIR, "data", "hotpot", "hotpot_dev_distractor_v1.json")
    hotpot_20_path = os.path.join(RAG_UPGRADE_DIR, "data", "hotpot", "hotpot_dev_distractor_20.json")

    if os.path.exists(mmlongbench_path):
        sample_json_list(mmlongbench_path, mmlongbench_20_path, 0.2)
    else:
        print(f"File not found: {mmlongbench_path}")

    if os.path.exists(hotpot_path):
        sample_hotpot_dict(hotpot_path, hotpot_20_path, 0.2)
    else:
        print(f"File not found: {hotpot_path}")
