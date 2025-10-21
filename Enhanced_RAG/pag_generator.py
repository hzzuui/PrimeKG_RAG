import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.neo4j_connect import Neo4jConnection

# === Neo4j 設定 ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "primekg666"   # ⚠️ 請改成你的密碼
NEO4J_DATABASE = "primekg"

# === 初始化 Neo4j 連線 ===
conn = Neo4jConnection(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)


def generate_pag(disease_name: str, limit: int = 5):
    query = f"""
    MATCH (d:disease)
    WHERE toLower(d.node_name) CONTAINS toLower($disease)
    MATCH (d)-[r]-(dr:drug)
    RETURN d.node_name AS disease,
           type(r) AS relation,
           dr.node_name AS drug,
           dr.description AS description,
           dr.mechanism_of_action AS moa,
           dr.indication AS indication,
           dr.atc_1 AS atc1,
           dr.atc_2 AS atc2,
           dr.atc_3 AS atc3,
           dr.atc_4 AS atc4
    LIMIT {limit}
    """

    results = conn.query(query, {"disease": disease_name})

    pag_list = []
    for r in results:
        disease = r.get("disease")
        drug = r.get("drug")
        relation = r.get("relation")
        moa = r.get("moa") or "unknown mechanism"
        indication = r.get("indication") or "unspecified indication"
        atc_codes = " > ".join([r.get("atc1") or "",
                                r.get("atc2") or "",
                                r.get("atc3") or "",
                                r.get("atc4") or ""]).strip(" >")
        desc = r.get("description") or ""

        # 根據關係生成句子
        if relation == "treated_by":
            text = f"{drug} is used to treat {disease}. "
        elif relation == "contraindication":
            text = f"{drug} is contraindicated for {disease}. "
        elif relation == "off-label use":
            text = f"{drug} is sometimes used off-label for {disease}. "
        else:
            text = f"{drug} is related to {disease} via {relation}. "

        if moa:
            text += f"It works by {moa}. "
        if atc_codes:
            text += f"Classified under ATC: {atc_codes}. "
        if indication and indication.lower() != "unspecified indication":
            text += f"Indication: {indication}. "

        pag_list.append({
            "disease": disease,
            "drug": drug,
            "relation": relation,
            "description": desc,
            "moa": moa,
            "atc_codes": atc_codes,
            "indication": indication,
            "pag_sentence": text.strip()
        })

    return pag_list

def generate_pag_drug(disease_name: str, conn, limit: int = 5):
    """
    Disease–Drug 層級的 PAG sentences
    """
    query = f"""
    MATCH (d:disease)
    WHERE toLower(d.node_name) CONTAINS toLower($disease)
    MATCH (d)-[r]-(dr:drug)
    RETURN d.node_name AS disease,
           type(r) AS relation,
           dr.node_name AS drug,
           dr.description AS description,
           dr.mechanism_of_action AS moa,
           dr.indication AS indication,
           dr.atc_1 AS atc1,
           dr.atc_2 AS atc2,
           dr.atc_3 AS atc3,
           dr.atc_4 AS atc4
    LIMIT {limit}
    """

    results = conn.query(query, {"disease": disease_name})
    pag_list = []

    for r in results:
        disease = r.get("disease")
        drug = r.get("drug")
        relation = r.get("relation")
        moa = r.get("moa") or "unknown mechanism"
        indication = r.get("indication") or "unspecified indication"
        atc_codes = " > ".join([r.get("atc1") or "",
                                r.get("atc2") or "",
                                r.get("atc3") or "",
                                r.get("atc4") or ""]).strip(" >")
        desc = r.get("description") or ""

        # 根據不同的關係生成句子
        if relation == "treated_by" or relation == "indication":
            text = f"{drug} is used to treat {disease}. "
        elif relation == "contraindication":
            text = f"{drug} is contraindicated for {disease}. "
        elif relation == "off-label use":
            text = f"{drug} is sometimes used off-label for {disease}. "
        else:
            text = f"{drug} is related to {disease} via {relation}. "

        if moa:
            text += f"It works by {moa}. "
        if atc_codes:
            text += f"Classified under ATC: {atc_codes}. "
        if indication and indication.lower() != "unspecified indication":
            text += f"Indication: {indication}. "

        pag_list.append({
            "disease": disease,
            "drug": drug,
            "relation": relation,
            "description": desc,
            "moa": moa,
            "atc_codes": atc_codes,
            "indication": indication,
            "pag_sentence": text.strip()
        })

    return pag_list

def generate_pag_with_genes(disease_name: str, conn, limit: int = 5):
    """
    Disease–Drug–Gene 層級的 PAG sentences
    """
    query = f"""
    MATCH (d:disease)
    WHERE toLower(d.node_name) CONTAINS toLower($disease)
    MATCH (d)<-[:indication]-(dr:drug)-[:drug_protein]->(g:gene__protein)
    RETURN d.node_name AS disease,
           dr.node_name AS drug,
           g.node_name AS gene,
           dr.description AS description,
           dr.mechanism_of_action AS moa,
           dr.indication AS indication,
           dr.atc_1 AS atc1,
           dr.atc_2 AS atc2,
           dr.atc_3 AS atc3,
           dr.atc_4 AS atc4
    LIMIT {limit}
    """

    results = conn.query(query, {"disease": disease_name})
    pag_list = []

    for r in results:
        disease = r.get("disease")
        drug = r.get("drug")
        gene = r.get("gene")
        moa = r.get("moa") or "unknown mechanism"
        indication = r.get("indication") or "unspecified indication"
        atc_codes = " > ".join([r.get("atc1") or "",
                                r.get("atc2") or "",
                                r.get("atc3") or "",
                                r.get("atc4") or ""]).strip(" >")
        desc = r.get("description") or ""

        # 生成句子
        text = f"{drug} is used to treat {disease}, targeting gene {gene}. "
        if moa:
            text += f"It works by {moa}. "
        if atc_codes:
            text += f"Classified under ATC: {atc_codes}. "
        if indication and indication.lower() != "unspecified indication":
            text += f"Indication: {indication}. "

        pag_list.append({
            "disease": disease,
            "drug": drug,
            "gene": gene,
            "description": desc,
            "moa": moa,
            "atc_codes": atc_codes,
            "indication": indication,
            "pag_sentence": text.strip()
        })

    return pag_list


if __name__ == "__main__":
    disease_query = "tinea pedis"   # ⚠️ 你可以改成任何疾病名稱
    pag_results = generate_pag(disease_query, limit=5)

    print(json.dumps(pag_results, indent=2, ensure_ascii=False))
