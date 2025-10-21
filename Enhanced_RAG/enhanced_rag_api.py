import json
import requests
import os
import csv
import sys
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# === 匯入 Neo4j 連線 & PAG 生成器 ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.neo4j_connect import Neo4jConnection
from pag_generator import generate_pag,generate_pag_drug,generate_pag_with_genes
# ========== 參數設定 ==========
MILVUS_COLLECTION = "collection_text"
QUERY_FILE = "data/primekg_queries_multihop.jsonl"
OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "gpt-oss:20b"  

RESULT_DIR = "results"
RESULT_JSON = os.path.join(RESULT_DIR, "enhanced_graph_result.json")
RESULT_CSV  = os.path.join(RESULT_DIR, "enhanced_graph_result.csv")

# ========== 初始化 ==========
print("🔧 連線到 Milvus ...")
connections.connect(alias="default", host="127.0.0.1", port="19530")

print("🔧 載入模型與 Milvus ...")
embedder = SentenceTransformer("all-mpnet-base-v2") # embedding model
collection = Collection(MILVUS_COLLECTION)

# 建立索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # IP = cosine similarity
    "params": {"nlist": 128}
}
try:
    collection.create_index(field_name="embedding", index_params=index_params)
    print("✅ 已建立索引 (embedding)")
except Exception as e:
    print(f"⚠️ 建立索引時跳過: {e}")

collection.load()
print(f"✅ Collection {MILVUS_COLLECTION} 已載入")

# ========== 工具函式 ==========

def search_milvus(query_text, top_k=5):
    """將 query 向量化，在 Milvus 檢索相似片段"""
    q_emb = embedder.encode(query_text).tolist()
    search_res = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "nprobe": 10}, 
        limit=top_k,
        output_fields=["source_name", "relation_type", "target_name"]
    )
    hits = [f"{hit.entity.get('source_name')} {hit.entity.get('relation_type')} {hit.entity.get('target_name')}"
            for hit in search_res[0]]
    return hits

def build_drug_gene_context(milvus_hits, conn: Neo4jConnection, limit=5):
    """Neo4j 擴展：疾病 → 藥物 → 基因"""
    context = []
    for hit in milvus_hits:
        entity = hit.split(" ")[0] if isinstance(hit, str) else hit.get("source_name", "")
        if not entity:
            continue
        query = f"""
        MATCH (d:disease)<-[:indication]-(drug:drug)-[:drug_protein]-(g:gene__protein)
        WHERE toLower(d.node_name) CONTAINS toLower("{entity}")
        RETURN d.node_name AS disease, drug.node_name AS drug, g.node_name AS gene
        LIMIT {limit}
        """
        records = conn.query(query)
        for r in records:
            context.append(f"{r['disease']} indication {r['drug']} drug_protein {r['gene']}")
    return context

def build_pag_context(milvus_hits, conn: Neo4jConnection, limit=2):
    """根據檢索的疾病，生成 PAG sentences"""
    pag_context = []
    for hit in milvus_hits:
        entity = hit.split(" ")[0] if isinstance(hit, str) else hit.get("source_name", "")
        if not entity:
            continue
        try:
            pag_results = generate_pag(entity, limit=limit)
            for pag in pag_results:
                pag_context.append(pag["pag_sentence"])
        except Exception as e:
            print(f"⚠️ PAG 生成失敗: {e}")
    return pag_context

def ask_llm(query_text, context, model=LLM_MODEL):
    """呼叫 Ollama API"""
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
    """Precision / Recall / F1"""
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
    print("🚀 開始 Enhanced-RAG 測試 ...")
    gold_queries = [json.loads(line) for line in open(QUERY_FILE, "r", encoding="utf-8")]
    all_results = []
    all_p, all_r, all_f1 = [], [], []

    conn = Neo4jConnection()

    for q in gold_queries:
        if q.get("type") != "multi-hop":
            continue
        question = q["question"]
        gold = [g.upper() for g in q["gold_answers"]]

        print(f"\n🔍 問題: {question}")

        # Step 1: Milvus 檢索
        milvus_context = search_milvus(question, top_k=5)
        print(f"  檢索到 {len(milvus_context)} 個片段")

        # Step 2: Neo4j 擴展
        graph_context = build_drug_gene_context(milvus_context, conn)
        print(f"  Neo4j 擴展到 {len(graph_context)} 個片段")

        # Step 3: PAG 生成
        # pag_context = build_pag_context(milvus_context, conn)
        pag_context = []
        for hit in milvus_context:
            entity = hit.split(" ")[0] if isinstance(hit, str) else hit.get("source_name", "")
            if not entity:
                continue

            # disease–drug 層級
            pag_drug = generate_pag_drug(entity, conn, limit=3)
            # disease–drug–gene 層級
            pag_gene = generate_pag_with_genes(entity, conn, limit=3)

            # 把句子抽出來
            for pag in (pag_drug + pag_gene):
                pag_context.append(pag["pag_sentence"])

        print(f"  PAG 生成 {len(pag_context)} 個句子")

        # Step 4: 合併 context
        full_context = milvus_context + graph_context + pag_context

        # Step 5: LLM 回答
        answer = ask_llm(question, full_context)

        # Step 6: 格式化回答
        pred = [a.strip().upper() for a in answer.split(",") if a.strip()]

        # Step 7: 計算指標
        p, r, f1 = compute_metrics(pred, gold)
        print(f"  Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

        # Step 8: 存結果
        result = {
            "question": question,
            "gold_answers": gold,
            "retrieved_context": full_context,
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
