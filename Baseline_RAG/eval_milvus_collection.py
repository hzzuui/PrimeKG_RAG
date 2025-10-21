import json
from collections import defaultdict

# 載入 gold answers
gold_file = "data/primekg_queries_multihop.jsonl"
gold_queries = [json.loads(line) for line in open(gold_file, "r", encoding="utf-8")]

# 載入 Milvus 匯出的 JSON
milvus_file = "data/collection_multi_hop.json"
milvus_data = json.load(open(milvus_file, "r", encoding="utf-8"))

retrieved = defaultdict(set)
for row in milvus_data:
    retrieved[row["query_entity"].lower()].add(row["target_gene"].lower())

def compute_metrics(pred, gold):
    pred_set, gold_set = set(pred), set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1

all_p, all_r, all_f1 = [], [], []

for q in gold_queries:
    qtext = q["question"]
    gold = [g.lower() for g in q["gold_answers"]]

    # 嘗試找到對應的 entity
    entity_key = None
    for k in retrieved.keys():
        if k in qtext.lower():
            entity_key = k
            break
    pred = retrieved.get(entity_key, [])

    p, r, f1 = compute_metrics(pred, gold)
    all_p.append(p); all_r.append(r); all_f1.append(f1)

    print(f"問題: {qtext}")
    print(f"  Gold 答案數: {len(gold)}, 預測數: {len(pred)}")
    print(f"  Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}\n")

print("=== 平均分數 ===")
print(f"Precision={sum(all_p)/len(all_p):.3f}, Recall={sum(all_r)/len(all_r):.3f}, F1={sum(all_f1)/len(all_f1):.3f}")
