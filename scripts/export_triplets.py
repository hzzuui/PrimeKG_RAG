from backend.neo4j_connect import Neo4jConnection

if __name__ == "__main__":
    conn = Neo4jConnection()
    triplets = conn.export_triplets()

    output_path = "triplets.tsv"
    with open(output_path, "w", encoding="utf-8") as f:
        # 寫入表頭（可選）
        f.write("head\trelation\ttail\n")
        for h, r, t in triplets:
            # 防止 None 值或 tab 錯誤
            h = str(h).replace('\t', ' ').strip()
            r = str(r).replace('\t', ' ').strip()
            t = str(t).replace('\t', ' ').strip()
            f.write(f"{h}\t{r}\t{t}\n")

    conn.close()
    print(f"✅ 成功匯出 {len(triplets)} 筆 triplets 到 {output_path}")
