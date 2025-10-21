import time
import json
import warnings
import sys
import socket
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from backend.neo4j_connect import Neo4jConnection
import os
import numpy as np

# ====== ✅ 忽略不必要的警告 ======
warnings.filterwarnings("ignore", category=FutureWarning)

# ====== ✅ Flask 初始化 ======
app = Flask(__name__)
CORS(app)

# ====== ✅ 設定參數 ======
MILVUS_HOST = "127.0.0.1"
COLLECTION_NAME = "collection_text"
COLLECTION_KGE = "collection_kge" 
EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_LLM_MODEL = "deepseek-r1:1.5b"  
DATA_DIR = "data"

# ====== ✅ 全域變數：延遲載入 Embedding 與 LLM ======
embedder = None
llm_cache = {}
load_lock = threading.Lock()  # 初始化鎖，避免 race condition

# ====== ✅ 載入 entity_to_vec (KGE) ======
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


# ====== ✅ 啟動檢查 ======
def startup_check():
    print("啟動檢查中...")
    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
        print(f"✅ 成功連接 Milvus at {MILVUS_HOST}:19530")
    except Exception as e:
        print(f"❌ 無法連接 Milvus：{e}")
        sys.exit(1)

    for col_name in [COLLECTION_NAME, COLLECTION_KGE]:
        if not utility.has_collection(col_name):
            print(f"❌ 找不到 Collection：{col_name}")
            sys.exit(1)
        col = Collection(col_name)
        print(f"✅ 找到 Collection：{col_name}，共 {col.num_entities} 筆資料")

    print("API 準備啟動...\n")

# ====== ✅ 路由區 ======
@app.route("/", methods=["GET"])
def index():
    return "✅ RAG API is running!"

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "embedder_loaded": embedder is not None,
        "llm_loaded": len(llm_cache) > 0,
        "kge_loaded": len(entity_to_vec) > 0
    })

@app.route("/neo_test", methods=["GET"])
def test_neo4j():
    try:
        conn = Neo4jConnection()
        result = conn.query("MATCH (n) RETURN COUNT(n) AS count")
        conn.close()
        count = result[0]["count"] if result else 0
        return jsonify({"neo4j_status": "connected", "total_nodes": count})
    except Exception as e:
        return jsonify({"neo4j_status": "error", "detail": str(e)}), 500

@app.route("/rag", methods=["POST"])
def rag():
    global embedder

    try:
        user_query = request.json.get("query")
        mode = request.json.get("mode", "同時顯示兩者")  # 預設為顯示兩者
        model_name = request.json.get("model", DEFAULT_LLM_MODEL)

        print(f"[DEBUG] Query received: {user_query}")
        print(f"[DEBUG] Mode received: {mode}")
        print(f"[DEBUG] Model selected: {model_name}")

        if not user_query:
            return jsonify({"error": "Query not provided"}), 400

        answer_rag = answer_no_rag = None

        # ====== 載入 LLM 模型 ======
        if model_name not in llm_cache:
            with load_lock:
                if model_name not in llm_cache:
                    print(f"[DEBUG] 載入 LLM 模型：{model_name}")
                    llm_cache[model_name] = OllamaLLM(model=model_name)
        llm = llm_cache[model_name]

        

        # ====== RAG 回應流程 ======
        if mode in ["RAG 回應", "同時顯示兩者"]:
            try:
                # 載入 embedder
                if embedder is None:
                    with load_lock:
                        if embedder is None:
                            print("[DEBUG] Loading embedder...")
                            embedder = SentenceTransformer(EMBED_MODEL)
                            print("[DEBUG] Embedder loaded")

                # 向量化
                query_vector = embedder.encode(user_query).tolist()
                print(f"[DEBUG] Query vector generated. Length: {len(query_vector)}")

                # === Step 1: Text 檢索，查詢 Milvus ===
                collection = Collection(COLLECTION_NAME)
                collection.load()
                print("[DEBUG] Milvus collection loaded")
                results = collection.search(
                    data=[query_vector],
                    anns_field="embedding",
                    param={"metric_type": "L2", "params": {"nprobe": 10}},
                    limit=5,
                    output_fields=["source_name", "relation_type", "target_name"]
                )
                print(f"[DEBUG] Milvus search returned {len(results[0]) if results else 0} results")

                # 建立 context
                context = ""
                related_entities = set()

                if results and results[0]:
                    for hit in results[0]:
                        s = hit.entity.get("source_name")
                        r = hit.entity.get("relation_type")
                        t = hit.entity.get("target_name")
                        context += f"{s} --[{r}]--> {t}\n"
                        related_entities.update([s, t])
                else:
                    print("[DEBUG] No context found from Milvus(Text)")

                


                # === Step 2: KGE 檢索 ===
                if related_entities and entity_to_vec:
                    print(f"[DEBUG] Doing KGE search for {len(related_entities)} entities...")
                    matched_vecs = [entity_to_vec[e] for e in related_entities if e in entity_to_vec]

                    # fallback
                    if not matched_vecs:
                        matched_vecs = [np.zeros(entity_embeddings.shape[1]).tolist()]

                    collection_kge = Collection(COLLECTION_KGE)
                    collection_kge.load()
                    kge_results = collection_kge.search(
                        data=matched_vecs,
                        anns_field="embedding",
                        param={"metric_type": "L2", "params": {"nprobe": 10}},
                        limit=3,
                        output_fields=["entity_name"]
                    )
                    for group in kge_results:
                        for r in group:
                            context += f"KGE Suggestion Entity: {r.entity.get('entity_name')}\n"
                            
                print(f"[DEBUG] Final context:\n{context if context else '[空白]'}")

                # 建立 prompt + 回應
                if context.strip():
                    # prompt_rag = f"""
                    # 你是一位知識圖譜專家。
                    # 以下包含兩部分檢索結果：
                    #     1. Text-based retrieval (Neo4j 三元組)
                    #     2. KGE-based retrieval (結構相似實體)

                    # 背景知識：{context}
                    # 問題：{user_query}
                    # 請用簡明扼要的方式作答。
                    # """

                    # prompt_rag = f"""
                    #     You are a biomedical knowledge graph expert.

                    #     Background knowledge:
                    #     {context}

                    #     Question: {user_query}

                    #     ### Instructions:
                    #     - Answer ONLY with the exact entity names from the knowledge graph (English).
                    #     - Provide them as a comma-separated list.
                    #     - Do NOT translate into Chinese.
                    #     - Do NOT add explanations or numbers.
                    #     """
                    prompt_rag = f"""
                    You are a biomedical knowledge graph expert.

                    Background knowledge:
                    {context}

                    Question: {user_query}

                    ### Instructions:
                    - Extract ONLY gene names from the background knowledge above.
                    - Do NOT output drugs or diseases.
                    - Respond ONLY in valid JSON, nothing else.

                    Output format:
                    {{
                    "genes": ["CYP3A4", "DRD2", "GRIN2B"]
                    }}
                    """
                    


                    print("[DEBUG] 生成 RAG 回應中...")
                    answer_rag = llm.invoke(prompt_rag)
                    # print("[DEBUG] LLM Answer (RAG):", answer_rag)
                    if not answer_rag.strip():
                        print("[WARNING] RAG 回應為空字串")
                else:
                    answer_rag = "(查無相關知識，無法使用 RAG 回應)"
            except Exception as e:
                print(f"[ERROR] RAG 回應失敗：{e}")
                answer_rag = f"(RAG 回應錯誤：{e})"

        # ====== no-RAG 回應流程 ======
        if mode in ["不使用 RAG 回應", "同時顯示兩者"]:
            try:
                prompt_no_rag = f"你是一位知識圖譜專家，請根據你自己的知識回答以下問題：{user_query}。請用簡明扼要的方式作答。"
                print("[DEBUG] 生成 No-RAG 回應中...")
                answer_no_rag = llm.invoke(prompt_no_rag)
                print("[DEBUG] LLM Answer (No-RAG):", answer_no_rag)
                if not answer_no_rag.strip():
                    print("[WARNING] No-RAG 回應為空字串")
            except Exception as e:
                print(f"[ERROR] No-RAG 回應失敗：{e}")
                answer_no_rag = f"(No-RAG 回應錯誤：{e})"

        return jsonify({
            "model": model_name,
            "mode": mode,
            "query": user_query,
            "context": context if "context" in locals() else None,
            "prompt_rag": prompt_rag if "prompt_rag" in locals() else None,
            "milvus_hits": [
                {
                    "source": hit.entity.get("source_name"),
                    "relation": hit.entity.get("relation_type"),
                    "target": hit.entity.get("target_name"),
                    "score": hit.distance
                }
                for hit in results[0]
            ] if results and results[0] else [],
            "answer_rag": answer_rag,
            "answer_no_rag": answer_no_rag
        })

    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

# ====== ✅ 顯示網址提示 ======
def show_access_urls():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("📡 透過以下網址訪問 API：")
    print(f" → 本機測試用：http://127.0.0.1:5000")
    print(f" → 區網設備用：http://{local_ip}:5000")

# ====== ✅ 啟動 Flask ======
if __name__ == "__main__":
    startup_check()
    show_access_urls()
    app.run(host="0.0.0.0", port=5000)
