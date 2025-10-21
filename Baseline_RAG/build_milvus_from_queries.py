import os
import json
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from backend.neo4j_connect import Neo4jConnection
from backend.milvus_connection import connect_milvus, create_collection

# === Step 1: 建立 Milvus Collection ===
def create_collection(name="collection_multi_hop", dim=768):
    if utility.has_collection(name):
        utility.drop_collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="query_entity", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="drug_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="target_gene", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="path_text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Multi-hop RAG collection (Disease→Drug→Gene)")
    collection = Collection(name, schema)
    return collection

# === Step 2: 從 Neo4j 抽取 multi-hop 子圖 ===
def extract_multi_hop(entity_name: str, conn: Neo4jConnection, limit=200):
    query = f"""
    MATCH (d:disease)<-[:indication]-(drug:drug)-[:drug_protein]-(g:gene__protein)
    WHERE d.node_name CONTAINS "{entity_name}"
    RETURN d.node_name AS disease, drug.node_name AS drug, g.node_name AS gene
    LIMIT {limit}
    """
    return conn.query(query)

# === Step 3: 插入 Milvus ===
def insert_embeddings(data, model, collection, entity_name, json_rows):
    rows = []
    for record in data:
        disease = record["disease"]
        drug = record["drug"]
        gene = record["gene"]

        path_text = f"{disease} treated_by {drug} targets {gene}"
        emb = model.encode(path_text).tolist()

        row = {
            "query_entity": disease,
            "drug_name": drug,
            "target_gene": gene,
            "path_text": path_text,
            "embedding": emb
        }
        rows.append(row)
        json_rows.append(row)  # 也存到 JSON

    if rows:
        collection.insert(rows)
        collection.flush()
    return len(rows)

# === Main ===
if __name__ == "__main__":
    queries_path = "data/primekg_queries_expanded.jsonl"
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if json.loads(line).get("type") == "multi-hop"]

    # 連線 Neo4j & Milvus
    conn = Neo4jConnection()
    connect_milvus()
    model = SentenceTransformer("all-mpnet-base-v2")

    collection_name = "collection_multi_hop"
    collection = create_collection(name=collection_name)
    print(f"✅ 建立新的 Milvus collection: {collection_name}")

    total_inserted = 0
    all_json_rows = []

    for q in queries:
        entity = q["question"].replace("治療", "").replace("的藥物作用於哪些基因？", "").strip() # e.g. "治療 mental disorder ..." → 取 disease
        print(f"🔍 抽取 multi-hop 子圖 for entity: {entity}")
        data = extract_multi_hop(entity, conn)
        inserted = insert_embeddings(data, model, collection, entity, all_json_rows)
        print(f"   → 插入 {inserted} 筆")
        total_inserted += inserted

    # === 存 JSON 備份 ===
    output_path = "data/collection_multi_hop.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_json_rows, f, ensure_ascii=False, indent=2)
    print(f"💾 已輸出 JSON 備份到 {output_path}")

    print(f"🎯 總共寫入 {total_inserted} 筆到 {collection_name}")
