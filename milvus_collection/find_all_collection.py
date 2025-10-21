from pymilvus import connections, utility

# 連線到 Milvus
connections.connect(alias="default", host="127.0.0.1", port="19530")

# 列出所有 collection
collections = utility.list_collections()
print("Existing collections:", collections)
