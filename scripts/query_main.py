# query_main.py 強化版，插入屬性推論模組
import sys
import os
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.milvus_connection import connect_milvus
from pymilvus import Collection
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import json
import numpy as np

from attribute_pipeline.attribute_reasoner import infer_attributes_from_context as infer_attributes # ✅ 新增

def load_entity_attributes_csv(file_path):
    entity_attr_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity = row["entity_id"]
            attributes = []
            for attr, value in row.items():
                if attr == "entity_id":
                    continue
                if value == "1":
                    attributes.append(attr)
            entity_attr_map[entity] = attributes
    return entity_attr_map

# === ✅ 載入 sanitized → 原始名稱對應 ===
DATA_DIR = "data"
name_map = {}
with open(os.path.join(DATA_DIR, "entity_name_map.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        safe_id, original_name = line.strip().split("\t")
        name_map[safe_id] = original_name

# === 載入 entity 向量與 safe_id（與 name_map 搭配使用） ===
entity_embeddings = np.load(os.path.join(DATA_DIR, "entity_embeddings.npy"))
with open(os.path.join(DATA_DIR, "entity_names.txt"), "r", encoding="utf-8") as f:
    safe_ids = [line.strip() for line in f if line.strip()]

entity_to_vec = {
    name_map[safe_id]: vec
    for safe_id, vec in zip(safe_ids, entity_embeddings)
    if safe_id in name_map
}

# === 載入實體屬性表格（原始 entity_attributes.csv） ===
entity_attr_map = load_entity_attributes_csv(os.path.join(DATA_DIR, "entity_attributes.csv"))

# === 載入 PAG 關聯圖 ===
with open(os.path.join(DATA_DIR, "attribute_graph.json"), "r", encoding="utf-8") as f:
    attribute_graph = json.load(f)


def prompt_find_connections(context: str, query: str) -> str:
    return f"""根據以下知識圖譜資料與屬性邏輯，探索與「{query}」可能有潛在關聯的基因、藥物或生物過程。
【資料】{context} 請用中文條列列出關聯實體，並簡要說明推測的原因或關係邏輯。"""


def hybrid_retrieval_and_llm(query, text_model, kge_collection, text_collection):
    query_text = text_model.encode(query).tolist()
    search_param = {"metric_type": "L2", "params": {"nprobe": 10}}

    text_results = text_collection.search(
        data=[query_text],
        anns_field="embedding",
        param=search_param,
        limit=3,
        output_fields=["source_name", "relation_type", "target_name"]
    )

    related_entities = set()
    for r in text_results[0]:
        related_entities.add(r.entity.get("source_name"))
        related_entities.add(r.entity.get("target_name"))

    matched_vecs = [entity_to_vec[name] for name in related_entities if name in entity_to_vec]
    if not matched_vecs:
        matched_vecs = [np.zeros(50).tolist()]

    kge_results = kge_collection.search(
        data=matched_vecs,
        anns_field="embedding",
        param=search_param,
        limit=3,
        output_fields=["entity_name"]
    )

    # === 🧠 新增屬性推論部分 ===
    reasoning_output = ""
    for entity in related_entities:
        if entity in entity_attr_map:
            known_attrs = entity_attr_map[entity]
            inferred_attrs = infer_attributes(known_attrs, attribute_graph)
            reasoning_output += f"\nEntity: {entity}\nKnown: {known_attrs}\nInferred: {inferred_attrs}\n"

    # === 拼接 context 給 LLM ===
    context = ""
    for r in text_results[0]:
        context += f"{r.entity.get('source_name')} -[{r.entity.get('relation_type')}]→ {r.entity.get('target_name')}\n"
    for group in kge_results:
        for r in group:
            context += f"KGE Suggestion Entity: {r.entity.get('entity_name')}\n"

    # ✅ 插入屬性推論補充
    context += f"\n[屬性邏輯推論]\n{reasoning_output}"

    prompt = prompt_find_connections(context, query)
    #llm = OllamaLLM(model="gemma:2b")
    llm = OllamaLLM(base_url="http://127.0.0.1:11435", model="gemma:2b")

    response = llm.invoke(f"{context}\n\n{prompt}")
    print("\n回答內容：\n")
    print(response)


if __name__ == "__main__":
    print("初始化模型與 Milvus")
    connect_milvus()
    text_model = SentenceTransformer("all-mpnet-base-v2")
    collection_text = Collection("collection_text")
    collection_kge = Collection("collection_kge")
    collection_text.load()
    collection_kge.load()

    query = input("請輸入查詢問題（如：stem cell regeneration in mammary gland）: ")
    hybrid_retrieval_and_llm(query, text_model, collection_kge, collection_text)
