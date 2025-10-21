# 建構 NLP 向量資料庫，僅在資料更新時執行 
# 建立 collection_text (單 hop)
# prepare_text_embeddings.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.neo4j_connect import Neo4jConnection
from backend.milvus_connection import connect_milvus, create_collection
from sentence_transformers import SentenceTransformer
import json

# === Step 1: 從 Neo4j 查詢節點(disease,gene,drug) ===
def extract_nodes():
    conn = Neo4jConnection()
    query = """
        MATCH path=(n)-[r:bioprocess_protein|pathway_protein|disease_protein|drug_effect|indication|phenotype_protein|drug_protein]-(m)
        RETURN n, r, m
    """
    data = conn.query(query)

    # === 保存原始 Neo4j Path 結果 ===
    raw_results = []
    for record in data:
        raw_results.append({
            "n": dict(record["n"]),   # 保留完整 node n
            "r": record["r"].type,    # 關係型別
            "m": dict(record["m"])    # 保留完整 node m
        })

    with open("neo4j_path.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=4)

    # === 保存精簡後的三元組 ===
    triples = []
    for record in data:
        n = record["n"]
        r = record["r"]
        m = record["m"]

        triples.append({
            "source_name": n["node_name"],
            "relation_type": r.type,
            "target_name": m["node_name"]
        })

    with open("neo4j_triples.json", "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=4)

    conn.close()
    print(f"✅ Saved {len(raw_results)} raw paths to neo4j_path.json")
    print(f"✅ Saved {len(triples)} triples to neo4j_triples.json")


# === Step 2: 將三元組轉向量並寫入 Milvus ===
def insert_text_embeddings(data, model, collection):
    vector_data = []
    seen = set()

    for item in data:
        source_name = item["source_name"]
        relation_type = item["relation_type"]
        target_name = item["target_name"]

        key = (source_name, relation_type, target_name)
        if key in seen:
            continue
        seen.add(key)

        # 🟢 Step 1: 建立自然語言化句子
        if relation_type == "drug_protein":
            text = f"The drug {source_name} targets the gene {target_name}."
        elif relation_type == "indication":
            text = f"The drug {source_name} is used to treat the disease {target_name}."
        elif relation_type == "disease_protein":
            text = f"The disease {source_name} is associated with the gene {target_name}."
        else:
            text = f"{source_name} {relation_type} {target_name}."

        # 🟢 Step 2: 轉向量
        embedding = model.encode(text).tolist()

        # 🟢 Step 3: 存進 collection
        row = {
            "source_name": source_name,
            "relation_type": relation_type,
            "target_name": target_name,
            "triple_text": text,
            "embedding": embedding
        }
        vector_data.append(row)

    # 插入 Milvus
    if vector_data:
        collection.insert(vector_data)
        collection.flush()
        print(f"✅ Inserted {len(vector_data)} text embeddings into Milvus.")
        print("當前 entities 數:", collection.num_entities)
    else:
        print("⚠️ 沒有要插入的資料！")

# === Main 執行區 ===
if __name__ == "__main__":
    print("step 1 : 連線至 Neo4j 並擷取資料...")
    extract_nodes()

    print("step 2 : 連線 Milvus 並建立 collection_text")
    connect_milvus()
    collection_text = create_collection(name="collection_text", dim=768)
    collection_text.load()

    print("step 3 : 向量化文字並寫入 Milvus")
    with open("neo4j_triples.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    model = SentenceTransformer("all-mpnet-base-v2")

    # 批次處理 + progress bar
    texts = []
    for item in data:
        if item["relation_type"] == "drug_protein":
            text = f"The drug {item['source_name']} targets the gene {item['target_name']}."
        elif item["relation_type"] == "indication":
            text = f"The drug {item['source_name']} is used to treat the disease {item['target_name']}."
        elif item["relation_type"] == "disease_protein":
            text = f"The disease {item['source_name']} is associated with the gene {item['target_name']}."
        else:
            text = f"{item['source_name']} {item['relation_type']} {item['target_name']}."
        texts.append((item["source_name"], item["relation_type"], item["target_name"], text))

    print(f"共 {len(texts)} 筆，開始向量化...")
    embeddings = model.encode([t[3] for t in texts], batch_size=64, show_progress_bar=True).tolist()

    vector_data = []
    for (source_name, relation_type, target_name, text), emb in zip(texts, embeddings):
        vector_data.append({
            "source_name": source_name,
            "relation_type": relation_type,
            "target_name": target_name,
            "triple_text": text,
            "embedding": emb
        })

    # 分批插入 Milvus
    print("開始插入 Milvus...")
    batch_size = 1000
    for i in range(0, len(vector_data), batch_size):
        batch = vector_data[i:i+batch_size]
        collection_text.insert(batch)
    collection_text.flush()

    print(f"✅ Inserted {len(vector_data)} text embeddings into Milvus.")
    print("當前 entities 數:", collection_text.num_entities)

