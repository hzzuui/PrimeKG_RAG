import torch
import numpy as np
import hashlib
from pykeen.triples import TriplesFactory
from pymilvus import Collection, utility
from backend.milvus_connection import connect_milvus, create_kge_collection
import os
KGE_COLLECTION = "collection_kge"

# 在 colab 上執行後下載 
DATA_DIR = "data"
EMBEDDING_FILE = os.path.join(DATA_DIR, "entity_embeddings.npy")
ENTITY_NAME_FILE = os.path.join(DATA_DIR, "entity_names.txt")


# 將 entity 名稱轉為 safe 格式（避免特殊符號、過長）
def safe_entity_id(entity_name):
    name = entity_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    return name if len(name) <= 255 else hashlib.md5(name.encode("utf-8")).hexdigest()

def load_kge_and_insert():
    print("Loading embeddings from local files...")

    # 載入資料
    embeddings = np.load(EMBEDDING_FILE)
    with open(ENTITY_NAME_FILE, "r", encoding="utf-8") as f:
        entity_names = [line.strip() for line in f.readlines()]
    print("Embedding shape:", embeddings.shape)
    assert len(entity_names) == embeddings.shape[0], "❌ entity 數量與 embedding 不符"


    # 建立映射檔案（可追溯原始名稱）
    with open(os.path.join(DATA_DIR, "entity_name_map.tsv"), "w", encoding="utf-8") as f:
        for name in entity_names:
            f.write(f"{safe_entity_id(name)}\t{name}\n")

    # 準備要插入 Milvus 的資料
    data_to_insert = [
        {"entity_name": str(safe_entity_id(name)), "embedding": embedding.tolist()}
        for name, embedding in zip(entity_names, embeddings)
    ]
    
    print("第一個 entity_name:", data_to_insert[0]["entity_name"])
    print("型別:", type(data_to_insert[0]["entity_name"]))
    
    # 插入資料
    collection = Collection(name=KGE_COLLECTION)
    collection.load()
    collection.insert(data_to_insert)
    print(f"Inserted {len(data_to_insert)} entity embeddings into '{KGE_COLLECTION}'")

if __name__ == "__main__":
    connect_milvus()
    if not utility.has_collection(KGE_COLLECTION):
        print(f"Creating collection '{KGE_COLLECTION}'...")
        create_kge_collection()
    else:
        print(f"Collection '{KGE_COLLECTION}' already exists")

    load_kge_and_insert()
