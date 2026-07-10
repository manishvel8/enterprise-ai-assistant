"""
seed-neo4j.py — Create Neo4j constraints and indexes for the project graph schema.

Run this once after Neo4j starts, before processing any documents.

Usage:
    python scripts/seed-neo4j.py

Requires:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j package not installed.")
    print("  pip install neo4j")
    sys.exit(1)


CONSTRAINTS = [
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
    "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (dec:Decision) REQUIRE dec.decision_id IS UNIQUE",
    "CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.action_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX document_file_name IF NOT EXISTS FOR (d:Document) ON (d.file_name)",
    "CREATE INDEX chunk_document_id IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX topic_name IF NOT EXISTS FOR (t:Topic) ON (t.name)",
]


def seed(driver):
    with driver.session() as session:
        print("\n--- Creating Constraints ---")
        for stmt in CONSTRAINTS:
            try:
                session.run(stmt)
                label = stmt.split("FOR")[1].strip().split(")")[0]
                print(f"  OK  {label}")
            except Exception as e:
                print(f"  SKIP (already exists or error): {e}")

        print("\n--- Creating Indexes ---")
        for stmt in INDEXES:
            try:
                session.run(stmt)
                label = stmt.split("FOR")[1].strip().split(")")[0]
                print(f"  OK  {label}")
            except Exception as e:
                print(f"  SKIP (already exists or error): {e}")

        print("\n--- Verifying Schema ---")
        result = session.run("SHOW CONSTRAINTS")
        constraints = list(result)
        print(f"  Total constraints: {len(constraints)}")

        result = session.run("SHOW INDEXES")
        indexes = [r for r in result if r["type"] != "LOOKUP"]
        print(f"  Total custom indexes: {len(indexes)}")


def main():
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connected successfully.")
    except Exception as e:
        print(f"ERROR: Could not connect to Neo4j: {e}")
        print("  Make sure Neo4j is running: docker compose up neo4j")
        sys.exit(1)

    try:
        seed(driver)
        print("\nNeo4j schema seeding complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
