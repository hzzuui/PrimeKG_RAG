import json
import re
import os
from typing import Tuple, List, Dict
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_ollama import OllamaLLM

# ========== 工具函式 ==========
def normalize_text(text: str) -> str:
    """簡單正規化，避免大小寫或標點差異"""
    return re.sub(r"\s+", " ", text.strip().lower())

def load_queries(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# --- fuzzy matching ---
def fuzzy_match(pred: str, gold: set, threshold=85) -> bool:
    return any(fuzz.ratio(pred, g) >= threshold for g in gold)

# --- embedding matching ---
embedder_eval = SentenceTransformer("all-MiniLM-L6-v2")

def embedding_match(pred: str, gold: set, threshold=0.75) -> bool:
    if not gold:
        return False
    pred_emb = embedder_eval.encode(pred, convert_to_tensor=True)
    gold_embs = embedder_eval.encode(list(gold), convert_to_tensor=True)
    sims = util.cos_sim(pred_emb, gold_embs)[0]
    return torch.any(sims >= threshold).item()

def compute_metrics(pred_set: set, gold_set: set, method="exact") -> Tuple[float, float, float]:
    """支援 exact / fuzzy / embedding 三種比對方式"""
    tp = fp = fn = 0

    for pred in pred_set:
        if method == "exact":
            matched = pred in gold_set
        elif method == "fuzzy":
            matched = fuzzy_match(pred, gold_set)
        elif method == "embedding":
            matched = embedding_match(pred, gold_set)
        else:
            raise ValueError(f"Unknown method: {method}")

        if matched:
            tp += 1
        else:
            fp += 1

    for gold in gold_set:
        if method == "exact":
            if gold not in pred_set:
                fn += 1
        elif method == "fuzzy":
            if not fuzzy_match(gold, pred_set):
                fn += 1
        elif method == "embedding":
            if not embedding_match(gold, pred_set):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

# --- Judge 模式 (LLM-as-a-Judge，本地呼叫) ---
judge_llm = OllamaLLM(model="qwen3:32b")  # 你可以改成 deepseek 或其他模型

def judge_eval(question: str, gold: List[str], model_answer: str) -> Tuple[float, float, float]:
    """使用 LLM 本地判斷 (不透過 API)"""
    prompt = f"""
    You are an impartial evaluation assistant.

    Your task:
    1. For each item in the model's answer, check if it matches any gold answer (semantic, EN/ZH OK).
    2. Count TP, FP, FN.
    3. Compute precision, recall, F1.
    4. Respond ONLY in JSON.

    Question:
    {question}

    Gold answers:
    {gold}

    Model's answer:
    {model_answer}

    Example output:
    {{"precision": 0.5, "recall": 0.4, "f1": 0.44}}
    """
    output = judge_llm.invoke(prompt).strip()
    print("=== Judge Raw Output ===")
    print(output)

    # 嘗試抓取 JSON 區塊
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        return 0.0, 0.0, 0.0

    try:
        judge_data = json.loads(match.group(0))
        return (
            float(judge_data.get("precision", 0.0)),
            float(judge_data.get("recall", 0.0)),
            float(judge_data.get("f1", 0.0)),
        )
    except Exception:
        return 0.0, 0.0, 0.0


# ========== 主程式 ==========
def evaluate_model(model_name: str, queries: List[Dict], max_samples: int = 20):
    metrics = {"exact": [], "fuzzy": [], "embedding": [], "judge": []}
    results = []

    for q in queries[:max_samples]:
        question = q["question"]
        gold = {normalize_text(a) for a in q["gold_answers"]}
        answer = q.get("model_answer", "")  # 直接用先前生成的答案
        answer_norm = normalize_text(answer)

        # === parsing ===
        predicted = {normalize_text(x) for x in re.split(r"[,\n;；。]|[0-9]+\.", answer_norm) if x.strip()}

        # === 三種比對方式 ===
        row_metrics = {}
        for method in ["exact", "fuzzy", "embedding"]:
            p, r, f1 = compute_metrics(predicted, gold, method=method)
            metrics[method].append((p, r, f1))
            row_metrics[f"{method}_precision"] = p
            row_metrics[f"{method}_recall"] = r
            row_metrics[f"{method}_f1"] = f1

        # === Judge 模式 (本地 LLM) ===
        p_j, r_j, f1_j = judge_eval(question, list(gold), answer)
        metrics["judge"].append((p_j, r_j, f1_j))
        row_metrics["judge_precision"] = p_j
        row_metrics["judge_recall"] = r_j
        row_metrics["judge_f1"] = f1_j

        results.append({
            "question": question,
            "gold_answers": list(gold),
            "model_answer": answer,
            "predicted_set": list(predicted),
            **row_metrics
        })

    # === 平均結果 ===
    avg_results = {}
    for method in metrics:
        if metrics[method]:
            avg_p = sum(m[0] for m in metrics[method]) / len(metrics[method])
            avg_r = sum(m[1] for m in metrics[method]) / len(metrics[method])
            avg_f1 = sum(m[2] for m in metrics[method]) / len(metrics[method])
        else:
            avg_p = avg_r = avg_f1 = 0.0
        avg_results[method] = (avg_p, avg_r, avg_f1)

    return avg_results, results


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    QUERIES_PATH = os.path.join(BASE_DIR, "data", "results_qwen.jsonl")  # ⚠️ 改成你存答案的檔案

    queries = load_queries(QUERIES_PATH)

    avg, res = evaluate_model("qwen3:14b", queries, max_samples=50)
    print("Qwen3:14b →", avg)

    out_path = os.path.join(BASE_DIR, "data", "evaluated_qwen.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 結果已存到 {out_path}")
