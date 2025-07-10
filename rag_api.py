import time
from flask import Flask, request, jsonify
from flask import Response, stream_with_context
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from flask import Response
import json
import warnings
import sys
import socket
from flask_cors import CORS
from backend.neo4j_connect import Neo4jConnection
import os

# ====== ✅ 忽略不必要的警告 ======
warnings.filterwarnings("ignore", category=FutureWarning)

# ====== ✅ Flask 初始化 ======
app = Flask(__name__)
CORS(app)  # ← 這行務必加上！


# ====== ✅ 設定參數 ======
MILVUS_HOST = "172.23.165.70"
COLLECTION_NAME = "primekg_rag_paths"
EMBED_MODEL = "all-mpnet-base-v2"
#LLM_MODEL = "gemma:2b"
LLM_MODEL = "qwen:0.5b"

# 向量模型與 LLM 延遲載入
embedder = None
llm = None

# ====== ✅ 啟動前檢查 ======
# Milvus 啟動檢查
def startup_check():
    print("啟動檢查中...")

    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
        print(f"成功連接 Milvus at {MILVUS_HOST}:19530")
    except Exception as e:
        print(f"無法連接 Milvus：{e}")
        sys.exit(1)

    if not utility.has_collection(COLLECTION_NAME):
        print(f"找不到 Collection：{COLLECTION_NAME}")
        sys.exit(1)

    col = Collection(COLLECTION_NAME)
    print(f"找到 Collection：{COLLECTION_NAME}，共 {col.num_entities} 筆資料")

    print("API 準備啟動...\n")

# ====== ✅ 主路由：RAG 查詢與 LLM 回應 ======
@app.route("/", methods=["GET"])
def index():
    return "RAG API is running!"


# 測試 Neo4j 是否成功連接並能查資料
@app.route("/neo_test", methods=["GET"])
def test_neo4j():
    try:
        conn = Neo4jConnection()
        test_query = "MATCH (n) RETURN COUNT(n) AS count"
        result = conn.query(test_query)
        conn.close()

        node_count = result[0]["count"] if result else 0
        return jsonify({"neo4j_status": "connected", "total_nodes": node_count})
    except Exception as e:
        return jsonify({"neo4j_status": "error", "detail": str(e)}), 500


@app.route("/rag", methods=["POST"])
def rag():
    global embedder, llm

    try:
        user_query = request.json.get("query")
        print(f"[DEBUG] Query received: {user_query}")

        if not user_query:
            return jsonify({"error": "Query not provided"}), 400

        if embedder is None:
            print("[DEBUG] Loading embedder...")
            embedder = SentenceTransformer(EMBED_MODEL)

        query_vector = embedder.encode(user_query).tolist()

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

        context = ""
        for hit in results[0]:
            s = hit.entity.get("source_name")
            r = hit.entity.get("relation_type")
            t = hit.entity.get("target_name")
            context += f"{s} --[{r}]--> {t}\n"

        prompt = f"""你是一位知識圖譜專家，請根據以下背景知識回答使用者問題：背景知識：{context}問題：{user_query}請用簡明扼要的方式作答。"""

        if llm is None:
            print(f"[DEBUG] Loading LLM: {LLM_MODEL}")
            llm = OllamaLLM(model=LLM_MODEL)

        answer = llm.invoke(prompt)
        print("[DEBUG] LLM 回應完成")

        return Response(json.dumps({"answer": answer}, ensure_ascii=False), mimetype="application/json")

    except Exception as e:
        print(f"[ERROR] {str(e)}")  # ✅ 印出錯誤細節
        return jsonify({"error": str(e)}), 500

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "data": [
            {
                "id":LLM_MODEL,
                "object": "model",
                "owned_by": "rag-api",
                "permission": [],
            }
        ],
        "object": "list"
    })

@app.route("/openai/v1/models", methods=["GET"])
def list_models_openai_format():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": LLM_MODEL,
                "object": "model",
                "owned_by": "rag-api",
                "permission": [],
            }
        ]
    })

'''
@app.route("/chat/completions", methods=["POST"])
def proxy_chat_completions():
    global embedder, llm

    try:
        body = request.get_json(force=True)  # ✅ 更保險地解析 JSON payload
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        user_query = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
        if not user_query:
            return jsonify({"error": "Query not provided"}), 400

        if embedder is None:
            print("[DEBUG] 載入向量模型...")
            embedder = SentenceTransformer(EMBED_MODEL)

        query_vector = embedder.encode(user_query).tolist()

        collection = Collection(COLLECTION_NAME)
        collection.load()

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["source_name", "relation_type", "target_name"]
        )

        context = ""
        for hit in results[0]:
            s = hit.entity.get("source_name")
            r = hit.entity.get("relation_type")
            t = hit.entity.get("target_name")
            context += f"{s} --[{r}]--> {t}\n"

        prompt = f"""你是一位知識圖譜專家，請根據以下背景知識回答使用者問題：背景知識：{context}問題：{user_query}請用簡明扼要的方式作答。"""

        if llm is None:
            print("[DEBUG] 載入 LLM 模型...")
            llm = OllamaLLM(model=LLM_MODEL)

        answer = llm.invoke(prompt)

        if stream:
            def generate():
                print("[DEBUG] Streaming 回應中...")
                for chunk in answer.split("\n"):
                    if chunk.strip():
                        yield 'data: ' + json.dumps({
                            "choices": [{
                                "delta": {"content": chunk + "\n"},
                                "index": 0,
                                "finish_reason": None  # ✅ 加上這個欄位
                            }],
                            "object": "chat.completion.chunk"
                        }, ensure_ascii=False) + '\n\n'
                yield 'data: ' + json.dumps({
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "object": "chat.completion.chunk"
                }) + '\n\n'
                yield 'data: [DONE]\n\n'



            return Response(stream_with_context(generate()), content_type='text/event-stream')

        else:
            return jsonify({
                "id": "chatcmpl-mockid",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer
                        },
                        "finish_reason": "stop"
                    }
                ]
            })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500
'''

# ====== ✅ 主路由：RAG 查詢與 LLM 回應 ======
@app.route("/openai/chat/completions", methods=["POST", "OPTIONS"])
def proxy_chat_completions():
    global embedder, llm
    if request.method == "OPTIONS":
        return '', 200  # ✅ 回傳 200 表示預檢成功
    try:
        # ✅ 印出完整 payload（可在 log 追蹤）
        body = request.get_json(force=True)
        print("[DEBUG] 接收到完整 payload:", body)

        # ✅ WebUI 常傳來冗餘欄位，這邊過濾僅取必要欄位
        messages = body.get("messages", [])
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "messages 欄位格式錯誤"}), 400

        # ✅ 強制關閉 stream 模式進行 debug（先跑通基本流程）
        stream = False

        # ✅ 取得最新一筆 user 查詢
        user_query = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
        if not user_query:
            return jsonify({"error": "找不到 user 提問內容"}), 400

        print(f"[DEBUG] 使用者查詢內容: {user_query}")

        # ✅ 載入向量模型
        if embedder is None:
            print("[DEBUG] 載入向量模型中...")
            embedder = SentenceTransformer(EMBED_MODEL)

        # ✅ 查詢 Neo4j 並轉換為向量
        print("[DEBUG] 查詢 Neo4j 並轉換為向量...")
        conn = Neo4jConnection()
        neo4j_query = """
        MATCH path = (n)-[:bioprocess_protein|pathway_protein|disease_protein|drug_effect|indication|phenotype_protein|drug_protein]-(m)
        WHERE n.node_name CONTAINS "stem cell" OR n.node_name CONTAINS "regenerative medicine"
        RETURN path
        """
        data = conn.query(neo4j_query)
        conn.close()
        print("🔌 Neo4j 連線已關閉")

        # ✅ 查詢結果過濾成上下文關係圖
        query_vector = embedder.encode(user_query).tolist()
        results = []
        seen = set()
        for i, item in enumerate(data):
            path = item.get("path", [])
            if len(path) != 3:
                continue
            source_node, relation_type, target_node = path
            key = (source_node["node_name"], relation_type, target_node["node_name"])
            if key in seen:
                continue
            seen.add(key)
            results.append(key)

        # ✅ 組合背景知識 context
        context = "\n".join([f"{s} --[{r}]--> {t}" for s, r, t in results[:5]])
        prompt = f"""你是一位知識圖譜專家，請根據以下背景知識回答使用者問題：背景知識：{context}問題：{user_query}請用簡明扼要的方式作答。"""

        # ✅ 載入 LLM 並推理
        if llm is None:
            print("[DEBUG] 載入 LLM 模型中...")
            llm = OllamaLLM(model=LLM_MODEL)

        print("[DEBUG] 呼叫 LLM 生成回答中...")
        answer = llm.invoke(prompt)

        # ✅ 非 streaming 模式下回傳標準格式
        return jsonify({
            "id": "chatcmpl-mockid",
            "object": "chat.completion",
            "created": int(time.time()),               # ✅ 加上 created timestamp
            "model": body.get("model", "unknown"),     # ✅ 加上 model 名稱
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {                                 # ✅ 加上 token 使用情況（暫填寫）
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

    except Exception as e:
        print(f"[❌ ERROR] ChatCompletions 內部錯誤：{str(e)}")
        return jsonify({"error": str(e)}), 500


# 列出可用網址（顯示 IP ）
def show_access_urls():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("透過以下網址訪問 API：")
    print(f" → 本機測試用： http://127.0.0.1:5000")
    print(f" → 區網設備用： http://{local_ip}:5000")


# ====== ✅ 啟動 Flask 伺服器 ======
if __name__ == "__main__":
    startup_check()
    show_access_urls()

    # 寫入 open-webui/.flask_ip
    flask_ip_path = os.path.join(os.path.dirname(__file__), "open-webui", ".flask_ip")
    local_ip = socket.gethostbyname(socket.gethostname())
    with open(flask_ip_path, "w") as f:
        f.write(local_ip)

    app.run(host="0.0.0.0", port=5000)
