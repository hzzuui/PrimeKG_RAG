import json
import re
import requests
import os
from typing import List, Dict

# ========== 工具函式 ==========
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def load_queries(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ========== 主程式：只呼叫 API，收集答案 ==========
def evaluate_model(model_name: str, queries: List[Dict], max_samples: int = 20):
    results = []

    for q in queries[:max_samples]:
        question = q["question"]

        # 呼叫 API (rag_api.py)
        resp = requests.post(
            "http://127.0.0.1:5000/rag",
            json={"query": question, "mode": "RAG 回應", "model": model_name}
        )

        if resp.status_code != 200:
            print(f"[ERROR] API call failed: {resp.text}")
            continue

        answer = resp.json().get("answer_rag", "")
        answer_norm = normalize_text(answer)

        # parsing 成 set
        predicted = {normalize_text(x) for x in re.split(r"[,\n;；。]|[0-9]+\.", answer_norm) if x.strip()}

        results.append({
            "question": question,
            "gold_answers": q["gold_answers"],   # 保留原始答案，但不比對
            "model_answer": answer,
            "predicted_set": list(predicted)
        })

    return results


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    QUERIES_PATH = os.path.join(BASE_DIR, "data", "primekg_queries_expanded.jsonl")

    queries = load_queries(QUERIES_PATH)
    queries = [q for q in queries if q.get("type") == "multi-hop"]  # 只取 multi-hop

    # Meditron
    res_meditron = evaluate_model("meditron:7b", queries, max_samples=50)
    out_meditron = os.path.join(BASE_DIR, "data", "results_meditron.jsonl")
    with open(out_meditron, "w", encoding="utf-8") as f:
        for r in res_meditron:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Qwen
    res_qwen = evaluate_model("qwen3:14b", queries, max_samples=50)
    out_qwen = os.path.join(BASE_DIR, "data", "results_qwen.jsonl")
    with open(out_qwen, "w", encoding="utf-8") as f:
        for r in res_qwen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 結果已存到 {out_meditron} 和 {out_qwen}")
