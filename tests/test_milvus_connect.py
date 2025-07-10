# test_milvus_connect.py

from pymilvus import connections

print("Connecting to Milvus on localhost:19530...")
connections.connect(alias="default", host="localhost", port="19530")
print("✅ Milvus 已成功連線")



'''
warnings.filterwarnings("ignore", category=UserWarning)

print("🔌 Connecting to Milvus on localhost:19530...")
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    secure=False,      # ✅ 明確關閉 TLS
    db_name="default"  # ✅ 指定預設資料庫名稱
)
print("✅ Connected to Milvus!")
'''

