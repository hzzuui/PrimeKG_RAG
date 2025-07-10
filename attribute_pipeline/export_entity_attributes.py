import pandas as pd
from collections import defaultdict
from backend.neo4j_connect import Neo4jConnection

OUTPUT_PATH = "data/entity_attributes.csv"

def fetch_entity_attributes():
    neo4j = Neo4jConnection()

    # 查詢所有節點間的關係與屬性類型（label）
    query = """
    MATCH (e)-[r]->(a)
    WHERE e.node_name IS NOT NULL AND a.node_name IS NOT NULL
    RETURN e.node_name AS entity_id, collect(DISTINCT head(labels(a))) AS attributes
    """
    results = neo4j.query(query)

    entity_attr_dict = defaultdict(set)
    for row in results:
        entity_id = row["entity_id"]
        attributes = row["attributes"]
        for attr in attributes:
            entity_attr_dict[entity_id].add(attr)

    # 所有出現過的屬性類型
    all_attrs = sorted({attr for attrs in entity_attr_dict.values() for attr in attrs})
    
    # 組成 binary 表格
    data = []
    for entity_id, attrs in entity_attr_dict.items():
        row = {attr: int(attr in attrs) for attr in all_attrs}
        row["entity_id"] = entity_id
        data.append(row)

    df = pd.DataFrame(data)
    df.set_index("entity_id", inplace=True)
    df.to_csv(OUTPUT_PATH)
    print(f"Entity-Attribute 表格已儲存至：{OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_entity_attributes()
