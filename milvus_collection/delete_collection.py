from pymilvus import connections, utility

# 1. 先建立連線
connections.connect(
    alias="default",
    host="127.0.0.1",   # 如果 Milvus 在 Docker，這裡要換成容器對外 IP
    port="19530"
)

# 2. 確認並刪除
if utility.has_collection("collection_text"):
    utility.drop_collection("collection_text")
    print("✅ collection_text 已刪除")
else:
    print("⚠️ collection_text 不存在")
