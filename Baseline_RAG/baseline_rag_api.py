import json
import requests
import os
import csv
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection


# ========== 參數設定 ==========
MILVUS_COLLECTION = "collection_text"
QUERY_FILE = "data/primekg_queries_multihop.jsonl"
OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "gpt-oss:20b"  

RESULT_DIR = "results"
RESULT_JSON = os.path.join(RESULT_DIR, "baseline_result.json")
RESULT_CSV  = os.path.join(RESULT_DIR, "baseline_result.csv")

# ========== 初始化 ==========
print("🔧 連線到 Milvus ...")
connections.connect(
    alias="default",
    host="127.0.0.1",   # 如果是 Docker 起的 Milvus，這裡要改成容器對外的 IP
    port="19530"
)

print("🔧 載入模型與 Milvus ...")
embedder = SentenceTransformer("all-mpnet-base-v2") # embedding model
collection = Collection(MILVUS_COLLECTION)
# 如果還沒有 index，建立一個
index_params = {
    "index_type": "IVF_FLAT",   # 你也可以改成 HNSW / AUTOINDEX
    "metric_type": "IP",        # IP = inner product (cosine 相似度)
    "params": {"nlist": 128}
}
try:
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ 已建立索引 (embedding)")
except Exception as e:
    print(f"⚠️ 建立索引時跳過: {e}")

# 再載入 collection
collection.load()
print(f"✅ Collection {MILVUS_COLLECTION} 已載入")

# ========== 工具函式 ==========
# 在指定的 Milvus collection 裡檢索相似向量。
def search_milvus(query_text, top_k=5):
    """將 query 向量化，並在 Milvus 檢索最相似的片段"""
    q_emb = embedder.encode(query_text).tolist()
    search_res = collection.search(
        data=[q_emb],
        anns_field="embedding", # 搜尋的基礎是 embedding 欄位。
        param={"metric_type": "IP", "nprobe": 10}, 
        limit=top_k,
        output_fields=["source_name", "relation_type", "target_name"]
    )
    # 把原本三個分開的欄位 → 合併成一段文字，提供給 LLM 當 context。
    hits = [f"{hit.entity.get('source_name')} {hit.entity.get('relation_type')} {hit.entity.get('target_name')}"  # 指定要取出的欄位。
        for hit in search_res[0]]

    return hits # 回傳檢索到的文字片段清單

def ask_llm(query_text, context, model=LLM_MODEL):
    """用 requests 呼叫 Ollama API"""
    prompt = f"""
你是一位生醫知識專家，以下是知識庫檢索到的相關資料：
{chr(10).join('- ' + c for c in context)}

問題：{query_text}
請直接列出基因清單，用逗號分隔。
"""
    response = requests.post(
        OLLAMA_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM API call failed: {response.status_code}, {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]

def compute_metrics(pred, gold):
    """計算 Precision / Recall / F1"""
    pred_set, gold_set = set(pred), set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1

# ========== 主程式 ==========
if __name__ == "__main__":
    print("🚀 開始 baseline RAG 測試 ...")
    gold_queries = [json.loads(line) for line in open(QUERY_FILE, "r", encoding="utf-8")]
    all_results = []
    all_p, all_r, all_f1 = [], [], []

    for q in gold_queries : 
        if q.get("type") != "multi-hop":
            continue
        question = q["question"]
        gold = [g.upper() for g in q["gold_answers"]]

        print(f"\n🔍 問題: {question}")

        # Step 1: Milvus 檢索
        context = search_milvus(question, top_k=5)
        print(f"  檢索到 {len(context)} 個片段")

        # Step 2: LLM 回答
        answer = ask_llm(question, context)
        print(f"  LLM 回答: {answer}")

        # Step 3: 格式化 LLM 回答（逗號分隔 → list）
        pred = [a.strip().upper() for a in answer.split(",") if a.strip()]

        # Step 4: 計算指標
        p, r, f1 = compute_metrics(pred, gold)
        print(f"  Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

        # Step 5: 存結果
        result = {
            "question": question,
            "gold_answers": gold,
            "retrieved_context": context,
            "llm_answer": answer,
            "pred_genes": pred,
            "precision": p,
            "recall": r,
            "f1": f1
        }
        all_results.append(result)

        

    # === 輸出 JSON ===
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # === 輸出 CSV ===
    with open(RESULT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

        # 總體平均
        avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
        avg_recall = sum(r["recall"] for r in all_results) / len(all_results)
        avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)

        writer.writerow({
            "question": "AVERAGE",
            "gold_answers": "-",
            "retrieved_context": "-",
            "llm_answer": "-",
            "pred_genes": "-",
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        })

    print(f"\n💾 已存檔: {RESULT_JSON}, {RESULT_CSV}")