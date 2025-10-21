from pymilvus import connections, Collection

# 1. 先連線到 Milvus
connections.connect(
    alias="default",
    host="127.0.0.1",  # 如果你 Milvus 在 Docker/WSL，可能要換成對應的 IP
    port="19530"
)

# 2. 指定要查看的 collection
col = Collection("collection_text")

# 3. 印出結構與資訊
print("Fields:", [f.name for f in col.schema.fields])
print("Schema:", col.schema)
print("Indexes:", col.indexes)
print("Entities count:", col.num_entities)

# 4. 查詢前五筆資料
if col.num_entities > 0:
    results = col.query(
        expr="",
        output_fields=["source_name", "relation_type", "target_name", "triple_text"],
        limit=5
    )
    print("\n=== 前五筆資料 ===")
    for r in results:
        print(r)
else:
    print("\n⚠️ 沒有資料可顯示")