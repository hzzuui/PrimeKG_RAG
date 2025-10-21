import json
from flask import Flask, request, jsonify
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
import numpy as np
import os
# ====== 初始化 ======
app = Flask(__name__)

MILVUS_HOST = "127.0.0.1"
COLLECTION_TEXT = "collection_text"
COLLECTION_KGE = "collection_kge"
EMBED_MODEL = "all-mpnet-base-v2"
LLM_MODEL = "qwen3:14b"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

embedder = SentenceTransformer(EMBED_MODEL)
llm = OllamaLLM(model=LLM_MODEL)

connections.connect(alias="default", host=MILVUS_HOST, port="19530")


# 載入 KGE 向量
print("🔄 載入 entity_to_vec...")
name_map = {}
with open(os.path.join(DATA_DIR, "entity_name_map.tsv"), "r", encoding="utf-8") as f:
    for line in f:
        safe_id, original_name = line.strip().split("\t")
        name_map[safe_id] = original_name

entity_embeddings = np.load(os.path.join(DATA_DIR, "entity_embeddings.npy"))
with open(os.path.join(DATA_DIR, "entity_names.txt"), "r", encoding="utf-8") as f:
    safe_ids = [line.strip() for line in f if line.strip()]

entity_to_vec = {
    name_map[safe_id]: vec
    for safe_id, vec in zip(safe_ids, entity_embeddings)
    if safe_id in name_map
}
print(f"✅ 已載入 entity_to_vec，共 {len(entity_to_vec)} 筆實體")




# ====== Route 1: Milvus 檢索 ======
@app.route("/debug/milvus", methods=["POST"])
def debug_milvus():
    query = request.json.get("query")
    q_vec = embedder.encode(query).tolist()

    collection = Collection(COLLECTION_TEXT)
    collection.load()
    results = collection.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 64}},
        limit=5,
        output_fields=["source_name", "relation_type", "target_name"]
    )

    hits = [
        {
            "source": h.entity.get("source_name"),
            "relation": h.entity.get("relation_type"),
            "target": h.entity.get("target_name"),
            "score": h.distance
        }
        for h in results[0]
    ]

    return jsonify({"query": query, "milvus_hits": hits})


# ====== Route 2: KGE 檢索 ======
@app.route("/debug/kge", methods=["POST"])
def debug_kge():
    query = request.json.get("query")
    q_vec = embedder.encode(query).tolist()

    # NLP 先抓實體
    collection = Collection(COLLECTION_TEXT)
    collection.load()
    results = collection.search(
        data=[q_vec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 64}},
        limit=5,
        output_fields=["source_name", "target_name"]
    )

    related_entities = set()
    for hit in results[0]:
        related_entities.update([hit.entity.get("source_name"), hit.entity.get("target_name")])

    matched_vecs = [entity_to_vec[e] for e in related_entities if e in entity_to_vec]
    if not matched_vecs:
        return jsonify({"query": query, "related_entities": list(related_entities), "kge_hits": []})

    collection_kge = Collection(COLLECTION_KGE)
    collection_kge.load()
    kge_results = collection_kge.search(
        data=matched_vecs,
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 64}},
        limit=3,
        output_fields=["entity_name"]
    )

    hits = []
    for group in kge_results:
        for r in group:
            hits.append({"entity": r.entity.get("entity_name"), "score": r.distance})

    return jsonify({"query": query, "related_entities": list(related_entities), "kge_hits": hits})


# ====== Route 3: LLM 測試 ======
@app.route("/debug/llm", methods=["POST"])
def debug_llm():
    query = request.json.get("query")
    context = request.json.get("context", "")

    prompt = f"""
    You are a biomedical knowledge graph expert.

    Background knowledge:
    {context}

    Question: {query}

    Answer concisely based only on the background knowledge.
    """
    answer = llm.invoke(prompt)

    return jsonify({"query": query, "context": context, "answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
