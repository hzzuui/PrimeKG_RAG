from neo4j import GraphDatabase

# 設定 Neo4j 連接資訊
NEO4J_URI = "bolt://localhost:7687"  # 確保 Neo4j 服務已啟動
#NEO4J_URI = "bolt://172.23.160.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "primekg666"  #自行更換密碼
NEO4J_DATABASE = "primekg"  

class Neo4jConnection:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        """ 關閉 Neo4j 連線 """
        if self._driver:
            self._driver.close()
            print("🔌 Neo4j 連線已關閉")

    def query(self, query, parameters=None):
        """ 執行 Cypher 查詢，返回結果 """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def export_triplets(self):
        query = """
        MATCH (h)-[r]->(t)
        WHERE h.node_name IS NOT NULL AND t.node_name IS NOT NULL
        RETURN h.node_name AS head, type(r) AS relation, t.node_name AS tail
        """
        results = self.query(query)

        triplets = []
        for record in results:
            try:
                head = record["head"]
                relation = record["relation"]
                tail = record["tail"]
                triplets.append((head, relation, tail))
            except KeyError:
                print("⚠️ 資料格式錯誤，跳過此筆紀錄")
                continue

        return triplets



