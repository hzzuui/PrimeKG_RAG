# 從已知的屬性集合中，透過屬性關聯圖（PAG）進行推論，補足可能遺漏的屬性
import pandas as pd
import networkx as nx
import json
import os

# 檔案路徑
ATTR_GRAPH_PATH = "data/attribute_graph.gml"
ENTITY_ATTR_PATH = "data/entity_attributes.csv"
CLUSTER_PATH = "data/attribute_clusters.json"
OUTPUT_PATH = "data/entity_attributes_inferred.csv"

def infer_attributes():
    # 讀取屬性圖（PAG）
    G = nx.read_gml(ATTR_GRAPH_PATH)

    # 讀取 entity 原始屬性表（0/1）
    df = pd.read_csv(ENTITY_ATTR_PATH)
    df.set_index("entity_id", inplace=True)

    # 讀取 root 屬性 cluster
    with open(CLUSTER_PATH, "r") as f:
        clusters = json.load(f)

    inferred_df = df.copy()

    for root_attr, cluster in clusters.items():
        print(f"根據 Root: {root_attr} 處理 Cluster: {cluster}")
        for entity_id, row in df.iterrows():
            known_attrs = [attr for attr in cluster if row[attr] == 1]
            inferred_attrs = set(known_attrs)

            # 遍歷每個已知屬性，透過圖找推論路徑
            for attr in known_attrs:
                descendants = nx.descendants(G, attr)
                for desc in descendants:
                    if desc in cluster:
                        inferred_attrs.add(desc)

            # 將推論出的屬性設為 1
            for attr in inferred_attrs:
                inferred_df.at[entity_id, attr] = 1

    inferred_df.reset_index(inplace=True)
    inferred_df.to_csv(OUTPUT_PATH, index=False)
    print(f"推論後屬性表已儲存至：{OUTPUT_PATH}")

# ✅ 即時推論：根據單一實體的 known_attrs + 屬性關聯圖推斷
def infer_attributes_from_context(known_attrs, attribute_graph):
    inferred = set()
    for attr in known_attrs:
        if attr in attribute_graph:
            inferred.update(attribute_graph[attr])
    return list(inferred - set(known_attrs))


if __name__ == "__main__":
    infer_attributes()
