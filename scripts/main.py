from backend.neo4j_connect import Neo4jConnection
from backend.milvus_connection import (
    connect_milvus, create_collection
)
import json
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from pymilvus import Collection

# === STEP 1: Neo4j 查詢再生醫療節點 ===
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

# === STEP 2: 建立 NLP 向量並寫入 Milvus（collection_text） ===
def insert_text_embeddings(data, model, collection):
    vector_data = []
    seen = set()
    for i, item in enumerate(data):
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

# === STEP 3: 雙椛查詢（NLP + KGE）後給 LLM ===
def hybrid_retrieval_and_llm(query, text_model, kge_collection, text_collection):
    query_text = text_model.encode(query).tolist()
    query_kge = query_text  # 簡化處理

    search_param = {"metric_type": "L2", "params": {"nprobe": 10}}

    text_results = text_collection.search(
        data=[query_text],
        anns_field="embedding",
        param=search_param,
        limit=3,
        output_fields=["source_name", "relation_type", "target_name"]
    )

    kge_results = kge_collection.search(
        data=[query_kge],
        anns_field="embedding",
        param=search_param,
        limit=3,
        output_fields=["entity_name"]
    )

    context = ""
    for r in text_results[0]:
        context += f"{r.entity.get('source_name')} -[{r.entity.get('relation_type')}]→ {r.entity.get('target_name')}\n"
    for r in kge_results[0]:
        context += f"KGE Suggestion Entity: {r.entity.get('entity_name')}\n"

    prompt = "請根據上述資料生成一份有關幹細胞再生在乳腺組織的應用報告"
    llm = OllamaLLM(model="gemma:2b")
    response = llm.invoke(f"{context}\n\n{prompt}")
    print(response)


# === 執行流程 ===
extract_nodes()
connect_milvus()

# 載入 Neo4j JSON
with open("neo4j_data.json", "r") as f:
    data = json.load(f)

# NLP 向量插入
text_model = SentenceTransformer("all-mpnet-base-v2")
collection_text = create_collection(name="collection_text", dim=768)
collection_text.load()
insert_text_embeddings(data, text_model, collection_text)

# KGE 向量插入 (已由 kge_train.py 預先訓練完成)
collection_kge = Collection(name="collection_kge")
collection_kge.load()

# 雙向量混合檢索
query = "stem cell regeneration in mammary gland"
hybrid_retrieval_and_llm(query, text_model, collection_kge, collection_text)
