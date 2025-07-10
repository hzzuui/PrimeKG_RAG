# 挑選 root 屬性並建立 cluster 群組
import networkx as nx
import json
import os

INPUT_GRAPH_PATH = "data/attribute_graph.gml"
OUTPUT_CLUSTER_PATH = "data/attribute_clusters.json"

# 可自訂要選幾個 Root 屬性
TOP_K_ROOTS = 3

def select_root_attributes(G, k=TOP_K_ROOTS):
    # 根據出度排序，出度高者影響力強
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
    roots = [node for node, _ in out_degrees[:k]]
    return roots

def build_attribute_clusters(G, roots):
    clusters = {}
    for root in roots:
        descendants = nx.descendants(G, root)
        clusters[root] = list(descendants | {root})  # 包含 root 本身
    return clusters

def main():
    if not os.path.exists(INPUT_GRAPH_PATH):
        print(f"找不到屬性關聯圖：{INPUT_GRAPH_PATH}")
        return

    # 載入 NetworkX 有向圖
    G = nx.read_gml(INPUT_GRAPH_PATH)

    # Step 1: 挑選 root 屬性
    root_attrs = select_root_attributes(G, k=TOP_K_ROOTS)
    print(f"挑選出的 Root 屬性：{root_attrs}")

    # Step 2: 建立 Cluster 群組
    clusters = build_attribute_clusters(G, root_attrs)

    # 儲存為 JSON 檔
    with open(OUTPUT_CLUSTER_PATH, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2)

    print(f"Cluster 群組已儲存至：{OUTPUT_CLUSTER_PATH}")

if __name__ == "__main__":
    main()
