import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# === 連線 Milvus ===
def load_collection(name="collection_multi_hop"):
    connections.connect("default", host="127.0.0.1", port="19530")
    return Collection(name)

# === 檢索 top-k 基因 ===
def retrieve_genes(collection, query_text, model, top_k=50):
    query_emb = model.encode(query_text).tolist()
    results = collection.search(
        data=[query_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["target_gene"]
    )
    # 取出基因名稱
    hits = results[0]
    return [hit.entity.get("target_gene") for hit in hits]

# === 計算 Precision / Recall / F1 ===
def compute_metrics(preds, golds):
    # 轉小寫 & 去掉空白避免 mismatch
    preds = set([p.lower().strip() for p in preds])
    golds = set([g.lower().strip() for g in golds])

    tp = len(preds & golds)
    fp = len(preds - golds)
    fn = len(golds - preds)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1, tp, fp, fn

# === Main ===
if __name__ == "__main__":
    # 載入 gold answer (只取 multi-hop 問題)
    queries_path = "data/primekg_queries_expanded.jsonl"
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if json.loads(line).get("type") == "multi-hop"]

    collection = load_collection("collection_multi_hop")
    model = SentenceTransformer("all-mpnet-base-v2")

    total_p, total_r, total_f1 = 0, 0, 0
    count = 0

    for q in queries:
        question = q["question"]
        gold_answers = q["gold_answers"]

        print(f"\n❓ 問題: {question}")
        print(f"🎯 Gold answers (部分): {gold_answers[:10]} ... 共 {len(gold_answers)}")

        # 用問題文字查
        preds = retrieve_genes(collection, question, model, top_k=50)

        precision, recall, f1, tp, fp, fn = compute_metrics(preds, gold_answers)

        print(f"🔍 檢索到 {len(preds)} 個基因，命中 {tp} 個")
        print(f"📊 Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        total_p += precision
        total_r += recall
        total_f1 += f1
        count += 1

    print("\n====== 總結 (multi-hop) ======")
    print(f"平均 Precision={total_p/count:.3f}")
    print(f"平均 Recall={total_r/count:.3f}")
    print(f"平均 F1={total_f1/count:.3f}")
