import random
import json
import os
from neo4j import GraphDatabase

# ====== Neo4j 連線設定 ======
NEO4J_URI = "bolt://localhost:7687"  # 確保 Neo4j 服務已啟動
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "primekg666"  
NEO4J_DATABASE = "primekg"  

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ====== 工具函式 ======
def run_query(cypher, params=None):
    """執行 Cypher 查詢，回傳結果 list"""
    with driver.session(database="primekg") as session:
        result = session.run(cypher, params or {})
        return [record[0] for record in result]

# ====== 抽樣節點 ======
def sample_nodes(label, limit=50):
    cypher = f"""
    MATCH (n:{label})
    RETURN n.node_name
    ORDER BY rand()
    LIMIT {limit}
    """
    return run_query(cypher)


# ====== 生成 Query + Gold Answer ======
def generate_fact_queries(diseases, drugs, genes):
    queries = []
    # Disease-Disease
    for d in random.sample(diseases, min(50, len(diseases))):
        cypher = """
        MATCH (d:disease {node_name:$disease})-[:disease_disease]->(d2:disease)
        RETURN d2.node_name
        """
        answers = run_query(cypher, {"disease": d})
        if len(answers) >= 2:
            queries.append({
                "type": "fact",
                "question": f"{d} 與哪些疾病相關？",
                "gold_answers": answers
            })
        if len(queries) >= 10:
            break

    # Disease-Phenotype Positive
    for d in random.sample(diseases, min(10, len(diseases))):
        cypher = """
        MATCH (d:disease {node_name:$disease})-[:disease_phenotype_positive]->(p:effect__phenotype)
        RETURN p.node_name
        """
        answers = run_query(cypher, {"disease": d})
        if answers:
            queries.append({
                "type": "fact",
                "question": f"{d} 會出現哪些正相關表現型（症狀）？",
                "gold_answers": answers
            })

    # Disease-Phenotype Negative
    for d in random.sample(diseases, min(10, len(diseases))):
        cypher = """
        MATCH (d:disease {node_name:$disease})-[:disease_phenotype_negative]->(p:effect__phenotype)
        RETURN p.node_name
        """
        answers = run_query(cypher, {"disease": d})
        if answers:
            queries.append({
                "type": "fact",
                "question": f"{d} 與哪些表現型呈負相關？",
                "gold_answers": answers
            })

    return queries

# A "list-type question" must have multiple answers
def generate_list_queries(diseases):
    queries = []
    for d in random.sample(diseases, min(10, len(diseases))):
        # find the positively associated phenotypes (symptoms) for a disease 
        cypher = """
        MATCH (d:disease {node_name:$disease})-[:disease_phenotype_positive]->(s:effect__phenotype)
        RETURN s.node_name
        """
        answers = run_query(cypher, {"disease": d})
        if answers :
            queries.append({
                "type": "list",
                "question": f"請列出 {d} 的常見症狀。",
                "gold_answers": answers
            })
    return queries
    
# 2-hop 查詢：Disease → Drug → Gene
def generate_multihop_queries(diseases, max_q=20):
    queries = []
    for d in random.sample(diseases, min(100, len(diseases))):  # 增加抽樣數量
        cypher = """
        MATCH (d:disease {node_name:$disease})<-[:indication]-(m:drug)-[:drug_protein]->(g:gene__protein)
        RETURN DISTINCT g.node_name
        """
        answers = run_query(cypher, {"disease": d})
        if len(answers) >= 1:   # 放寬條件
            queries.append({
                "type": "multi-hop",
                "hops": 2,
                "question": f"治療 {d} 的藥物作用於哪些基因？",
                "gold_answers": answers
            })
        if len(queries) >= max_q:   # 控制最多輸出數量
            break
    return queries


# ====== 主程式 ======
def main():
    diseases = sample_nodes("disease", 200)
    drugs = sample_nodes("drug", 200)
    genes = sample_nodes("gene__protein", 200)

    all_queries = []
    all_queries.extend(generate_fact_queries(diseases, drugs, genes))
    all_queries.extend(generate_list_queries(diseases))
    all_queries.extend(generate_multihop_queries(diseases)) # multi-hop

    # 只保留 100 題
    random.shuffle(all_queries)
    all_queries = all_queries[:100]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "data", "primekg_queries.jsonl")

    
    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"已生成 {len(all_queries)} 個 query，輸出到 primekg_queries.jsonl")

if __name__ == "__main__":
    
    # testing function 
    diseases = sample_nodes("disease", 10)
    drugs = sample_nodes("drug", 10)
    genes = sample_nodes("gene__protein", 10)

    print("✅ sample_nodes 測試：", diseases[:5])

    fact_q = generate_fact_queries(diseases, drugs, genes)
    print("✅ generate_fact_queries 測試：", json.dumps(fact_q[:2], ensure_ascii=False, indent=2))

    list_q = generate_list_queries(diseases)
    print("✅ generate_list_queries 測試：", json.dumps(list_q[:2], ensure_ascii=False, indent=2))

    # 執行主程式
    main()

