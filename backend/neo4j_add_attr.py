import pandas as pd
from neo4j_connect import Neo4jConnection

# === Neo4j 連線設定 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "primekg666"   # ⚠️ 改成你自己的密碼
NEO4J_DATABASE = "primekg"

# === 讀取 disease_features.csv ===
disease_df = pd.read_csv("data/drug_features.csv")

# === 建立 Neo4j 連線 ===
conn = Neo4jConnection(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# === 選擇要加入的欄位 ===
columns_to_add = [
    "description",
    "half_life",
    "indication",
    "mechanism_of_action",
    "protein_binding",
    "pharmacodynamics",
    "state",
    "atc_1",
    "atc_2",
    "atc_3",
    "atc_4",
    "category",
    "group",
    "pathway",
    "molecular_weight",
    "tpsa",
    "clogp"
]

# === 更新 Neo4j 節點 ===
for _, row in disease_df.iterrows():
    set_clauses = []
    params = {"id": str(row["node_index"])}   # ⚠️ 改成 str，因為 DB 裡是字串

    for col in columns_to_add:
        if col in row and pd.notna(row[col]):
            set_clauses.append(f"d.{col} = ${col}")
            params[col] = str(row[col])  # 避免 NaN

    if set_clauses:
        query = f"""
        MATCH (d:drug {{node_index: $id}})
        SET {", ".join(set_clauses)}
        """
        conn.query(query, params)

print("✅ drug 屬性更新完成！")


# === 讀取 disease_features.csv ===
disease_df = pd.read_csv("data/disease_features.csv")

# === 建立 Neo4j 連線 ===
conn = Neo4jConnection(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

# === 選擇要加入的欄位 ===
columns_to_add = [
    "mondo_definition",
    "umls_description",
    "orphanet_definition",
    "orphanet_prevalence",
    "orphanet_epidemiology",
    "orphanet_clinical_description",
    "orphanet_management_and_treatment",
    "mayo_symptoms",
    "mayo_causes",
    "mayo_risk_factors",
    "mayo_complications",
    "mayo_prevention",
    "mayo_see_doc"
]

# === 更新 Neo4j 節點 ===
for _, row in disease_df.iterrows():
    set_clauses = []
    params = {"id": str(row["node_index"])}   # ⚠️ 改成 str，因為 DB 裡是字串

    for col in columns_to_add:
        if col in row and pd.notna(row[col]):
            set_clauses.append(f"d.{col} = ${col}")
            params[col] = str(row[col])  # 避免 NaN

    if set_clauses:
        query = f"""
        MATCH (d:disease {{node_index: $id}})
        SET {", ".join(set_clauses)}
        """
        conn.query(query, params)

print("✅ Disease 屬性更新完成！")
