# 建構 NLP 向量資料庫，僅在資料更新時執行
# prepare_text_embeddings.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.neo4j_connect import Neo4jConnection
from backend.milvus_connection import connect_milvus, create_collection
from sentence_transformers import SentenceTransformer
import json

# === Step 1: 從 Neo4j 查詢再生醫療節點 ===
def extract_nodes():
    conn = Neo4jConnection()
    query = """
    MATCH path = (n)-[:bioprocess_protein|pathway_protein|disease_protein|drug_effect|indication|phenotype_protein|drug_protein]-(m)
    WHERE n.node_name CONTAINS "stem cell" OR n.node_name CONTAINS "regenerative medicine"
    RETURN path
    """
    data = conn.query(query)
    with open("neo4j_data.json", "w") as f:
        json.dump([record.data() for record in data], f, indent=4)
    conn.close()
    print(f"Extracted {len(data)} paths from Neo4j.")

# === Step 2: 將三元組轉向量並寫入 Milvus ===
def insert_text_embeddings(data, model, collection):
    vector_data = []
    seen = set()
    for item in data:
        path = item.get("path", [])
        if len(path) != 3:
            continue
        source_node = path[0]
        relation_type = path[1]
        target_node = path[2]

        key = (source_node["node_name"], relation_type, target_node["node_name"])
        if key in seen:
            continue
        seen.add(key)

        text = f"{source_node['node_name']} {relation_type} {target_node['node_name']}"
        embedding = model.encode(text).tolist()

        row = {
            "source_name": source_node["node_name"],
            "relation_type": relation_type,
            "target_name": target_node["node_name"],
            "embedding": embedding
        }
        vector_data.append(row)

    collection.insert(vector_data)
    print(f"Inserted {len(vector_data)} text embeddings into Milvus.")

# === Main 執行區 ===
if __name__ == "__main__":
    print("step 1 : 連線至 Neo4j 並擷取資料...")
    extract_nodes()

    print("step 2 : 連線 Milvus 並建立 collection_text")
    connect_milvus()
    collection_text = create_collection(name="collection_text", dim=768)
    collection_text.load()

    print("step 3 : 向量化文字並寫入 Milvus")
    with open("neo4j_data.json", "r") as f:
        data = json.load(f)
    model = SentenceTransformer("all-mpnet-base-v2")
    insert_text_embeddings(data, model, collection_text)

