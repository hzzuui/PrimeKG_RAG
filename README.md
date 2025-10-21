# PrimeKG-RAG 系統：結合 Neo4j、Milvus、Ollama 與 Streamlit UI 的智慧知識問答平台

本專案旨在打造一套結合 **PrimeKG 知識圖譜**、**RAG（檢索增強生成）架構**、**本地語言模型（Ollama）**、**Milvus 向量資料庫** 與 **Neo4j 圖資料庫** 的智慧問答平台，並透過 **Streamlit UI 介面**，讓使用者能以自然語言發問，並獲得結合知識圖譜的語意答案。

---

## 專案特色

- **整合 PrimeKG 生醫知識圖譜**：以 Neo4j 儲存節點與關係資訊。
- **支援語意檢索的 RAG 架構**：透過 Sentence-BERT 或 KGE 生成向量，儲存在 Milvus 中。
- **Ollama 本地 LLM 語言模型**：如 `gemma`、`qwen` 等，用於生成自然語言答案。
- **Streamlit UI 介面**：可視化互動式提問體驗，方便非技術使用者使用。
- **結合 KGE 嵌入（Knowledge Graph Embedding）**：提升檢索與語意理解精度。

---

## 架構總覽 - 使用 streamlit UI

使用者 → Streamlit Chat UI
↓
🔗 HTTP POST /rag → rag_api.py (python -m rag_api)
↓
Embedding: SentenceTransformer(user query)
↓
🔍 檢索：Milvus（NLP 向量）+ Neo4j（圖譜路徑）
↓
Prompt 結合語意 + 結構 + KGE （可擴充）
↓
呼叫 Ollama（如 gemma:2b）
↓
回傳結果給 Streamlit 顯示

## 技術棧

| 模組         | 技術 / 工具                                          |
| ------------ | ---------------------------------------------------- |
| 語言模型     | [Ollama](https://ollama.com/) + 本地 LLM（如 gemma） |
| 向量資料庫   | [Milvus](https://milvus.io/)                         |
| 知識圖譜     | [Neo4j](https://neo4j.com/)                          |
| 向量嵌入     | Sentence-Transformers / KGE（TransE）                |
| 後端 API     | Flask / FastAPI                                      |
| 前端介面     | [Streamlit UI ]                                      |
| 知識圖譜來源 | [PrimeKG](https://github.com/mims-harvard/PrimeKG)   |
| 向量計算     | `sentence-transformers` / `pykeen` (KGE)             |

---

# RAG

## 安裝與啟動

1. **clone 專案**

```bash
git clone https://github.com/hzzuui/PrimeKG_RAG.git

```

2. **建立虛擬環境與安裝依賴**

```bash
pip install -r requirements.txt

```

3. **啟動 Neo4j 與 Milvus**

4. **載入 PrimeKG 到 Neo4j**

```bash
python scripts/load_primekg_to_neo4j.py
```

5. **訓練並載入 KGE 模型**

```bash
python scripts/train_kge_embeddings.py
```

6. **寫入向量到 Milvus**

```bash
python scripts/load_embeddings_to_milvus.py
```

7. **啟動 Flask API（RAG + Neo4j + Milvus）**

```bash
python rag_api.py
```

# 結合 KGE + NLP

## 環境架構概覽

- Docker Desktop : 負責啟動 Milvus
  - docker-compose up -d
- WSL : 執行 ollama serve

  - 啟用輕量 wsl (Ubuntu-22.04)，用來啟用 ollama 服務
  - ollama serve
  - ollama run gemma:2b

- Neo4j Desktop : 啟動圖資料庫(PrimeKG)

1. **Milvus 啟動與連線測試**

Step 1: 容器清理 (可選)
若使用 powershell 環境，需先啟動 docker desktop

```bash
docker compose down -v
docker volume prune -f
docker rm -f milvus-standalone 2>$null
```

以上指令用於移除先前未關閉乾淨的容器與掛載 volume。

Step 2: 啟動 Milvus

```bash
docker-compose up -d
docker compose up -d --build

```

透過以下指令，確保 milvus docker 的狀態皆為 healthy

```bash
docker ps
```

Step 3 : 測試 milvus 是否能成功連線

執行 test_milvus_connect.py

```bash
python tests/test_milvus_connect.py
```

成功連線會回傳 : Milvus 已成功連線

2. **匯出知識圖譜三元組資料**
   透過 export_triplets.py 從 Neo4j 中匯出三元組（triplets）供 KGE 訓練使用。產出 triplets.tsv。接著到 colab 進行 KGE 訓練

3. **KGE 建置**

執行 kge_train.py

```bash
python -m scripts.kge_train
```

Milvus 向量庫建置成功 : Inserted 129262 entity embeddings into 'collection_kge'
並會產生 entity_name_map.tsv

4. **建構 NLP 向量資料庫**
   用於從 Neo4j 中擷取與「再生醫療」有關的節點關係，並將它們轉換成文字向量（text embeddings）後，寫入 Milvus 向量資料庫，以供後續檢索式生成（RAG）使用。
   需先到 Neo4j DESKTOP 啟動圖資料庫，再執行 prepare_text_embeddings.py

```bash
python scripts/prepare_text_embeddings.py
```

建置成功回傳 : Inserted 488 text embeddings into Milvus.

執行條件與時機：

- 第一次建置 NLP 向量資料庫時執行一次
- Neo4j 資料更新後重新執行

5. **執行查詢主程式 query_main.py**

step 1: 啟動本地語言模型服務（Ollama），先開啟終端機(wsl 環境)，依序執行以下指令：

```bash
ollama serve
ollama run gemma:2b
```

用來啟動 Ollama 的後端服務，並載入 gemma:2b 模型，供主程式呼叫回答問題。

step 2:執行 query_main.py，開啟另一個終端機視窗，執行以下指令：

```bash
python scripts/query_main.py
```

系統將會進行以下流程：

- 文字語意查詢（NLP Embeddings）: 使用 SentenceTransformer 模型（all-mpnet-base-v2）將使用者輸入的問題轉為向量，並在 Milvus 的文字向量資料庫中查找語意上最相似的三元組資料（如：某細胞涉及的生物過程或藥物作用）。
- 結構知識查詢（KGE Embeddings）: 從 NLP 查詢結果中擷取相關實體名稱，並利用事先訓練好的知識圖譯碼（Knowledge Graph Embeddings），找出圖譜中結構上相似或潛在相關的其他實體。
- 生成式回答（LLM：Ollama 上的 Gemma:2b）: 將 NLP 查詢與 KGE 查詢的結果整理成文字 context，搭配設計好的 prompt，交由本地語言模型推論可能的關聯關係，並以條列式方式生成具邏輯性的中文回答。

這樣即可實現一個融合結構檢索與語言理解的智慧問答流程。若想查詢「例如：stem cell regeneration in mammary gland」之類問題，可依提示輸入關鍵字

# 新增推導

## Step1

從 Neo4j 的知識圖譜中萃取出每個實體（entity）擁有哪些屬性（attribute），並將這些資訊轉換為二進位表格（binary matrix），儲存成 entity_attributes.csv，供後續邏輯推論分析用。

需先啟動 Neo4j Desktop，再執行以下指令

```bash
python -m attribute_pipeline.export_entity_attributesin
```

輸出：entity_attributes.csv

## Step 2：挖掘屬性間的 A → B 關係（implication mining）

```bash
python -m attribute_pipeline.mine_attribute_implications
```

輸出：attr_implications.csv
(後續可比較不同關聯規則條件)

## Step 3：建構屬性關聯圖（PAG）

利用前一步挖掘出來的屬性推論規則（如 A → B），建立一張 可查詢與可視化的屬性邏輯關聯圖，用於 強化 RAG 系統的 context 補全能力

```bash
python -m attribute_pipeline.build_attribute_graph
```

## Step 4：產生屬性群（cluster）與 node splitting (xᵢ, bᵢ, zᵢ)

```bash
python -m attribute_pipeline.cluster_by_root_attribute
```

輸出：attribute_clusters.json
目的：

## Step 5：進行屬性推理（Attribute Reasoning）

從某個實體的部分屬性集合出發，透過屬性關聯圖（PAG）推理出它可能還有哪些屬性（例如 A → B → C）。

```bash
python -m attribute_pipeline.attribute_reasoner
```

輸出：entity_attributes_inferred.csv (和 step 1 產出的 entity_attributes.csv 進行比較)

## Step 6：推論評估（Attribute Inference Evaluation）

衡量 Step 5 推論的準確性與補全效果，評估指標如下 :

- True Positive (TP)：原本是 0，推論後是 1，且在實際中是正確的。
- False Positive (FP)：原本是 0，推論為 1，但實際不應該是。
- False Negative (FN)：原本是 0，推論仍是 0，但實際應該是 1。
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 × (Precision × Recall) / (Precision + Recall)

```bash
python -m attribute_pipeline.evaluate_inference
```

## Step 7：轉換 attribute_graph.pkl 為 attribute_graph.json，提供給 query_main.py 使用

```bash
python scripts/convert_attribute_graph.py
```

### 使用 chatui.py 來建立一個 ChatBot 的介面

```bash
python -m streamlit run chatui.py
```

```bash
python -m streamlit run chatui.py
```
