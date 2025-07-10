from neo4j import GraphDatabase

uri = "bolt://172.23.165.70:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "200103008"))
with driver.session(database="primekg") as session:
    result = session.run("MATCH (n) RETURN count(n)")
    print(result.single())