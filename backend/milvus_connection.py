from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections

# Milvus 伺服器連線資訊
#MILVUS_HOST = "172.23.165.70"  # 本機 WSL IP
#MILVUS_HOST = "host.docker.internal" 
MILVUS_HOST = "localhost" 
MILVUS_PORT = "19530"
COLLECTION_NAME = "primekg_rag_paths"

def connect_milvus():
    """ 連線 Milvus 伺服器 """
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Milvus 連線成功")

def check_connection():
    """ 檢查 Milvus 連線狀態 """
    status = connections.has_connection("default")
    print("Milvus 連接狀態:", status)
    return status

def create_collection(name=COLLECTION_NAME, dim=768):
    """
    自製化建立 Milvus Collection
    - name: Collection 名稱
    - dim: 向量縮小維度 (e.g., 768 for NLP, 50 for KGE)
    """
    if name == "collection_text":
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="relation_type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="target_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="triple_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
    elif name == "collection_kge":
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
    else:
        # fallback: 預設使用原有 schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="relation_type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="target_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="target_type", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="target_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]

    schema = CollectionSchema(fields, description=f"Milvus Collection: {name}")
    collection = Collection(name=name, schema=schema)
    print(f"Milvus Collection '{name}' 已建立")

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )
    return collection

def get_collection():
    """ 取得 Milvus Collection，如果不存在則建立 """
    try:
        collection = Collection(name=COLLECTION_NAME)
        print(f"📌 Collection '{COLLECTION_NAME}' 已存在，直接使用")
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' 不存在，正在建立...")
        create_collection()
        collection = Collection(name=COLLECTION_NAME)
    return collection

def close_milvus():
    """ 關閉 Milvus 連線 """
    connections.disconnect("default")
    print("🔌 Milvus 連線已關閉")


def create_kge_collection(name="collection_kge", dim=50):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="entity_name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="KGE Entity Embeddings")
    collection = Collection(name=name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
    )
    return collection

def get_entity_id_name_map(self) -> dict:
        query = """
        MATCH (n)
        WHERE exists(n.name) AND exists(n.id)
        RETURN n.id AS id, n.name AS name
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {str(r["id"]): r["name"] for r in result}
        
        
# 測試執行
if __name__ == "__main__":
    connect_milvus()
    get_collection()
    close_milvus()
    