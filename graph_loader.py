import os
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_schema(self):
        """
        Creates necessary constraints and indexes.
        """
        with self.driver.session() as session:
            # Constraint: Paper ID must be unique
            session.run("CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            logger.info("Schema constraints set up.")
            
            # Note: Vector indexes usually require specific configuration depending on Neo4j version
            # For Neo4j 5.x+:
            try:
                session.run("""
                    CREATE VECTOR INDEX paper_problem_embedding IF NOT EXISTS
                    FOR (p:Paper) ON (p.embedding_problem)
                    OPTIONS {indexConfig: {
                     `vector.dimensions`: 384,
                     `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Vector index 'paper_problem_embedding' created.")
            except Exception as e:
                logger.warning(f"Could not create vector index (might be older Neo4j version): {e}")

    def add_paper(self, paper_data):
        """
        Adds or updates a Paper node.
        paper_data: dict containing 'id', 'title', 'year', 'problem_statement', etc.
        """
        query = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.year = $year,
            p.venue = $venue,
            p.paper_type = $paper_type,
            p.problem_statement = $problem_statement,
            p.core_method = $core_method,
            p.key_findings = $key_findings,
            p.embedding_problem = $embedding_problem,
            p.embedding_method = $embedding_method
        RETURN p.id
        """
        with self.driver.session() as session:
            result = session.run(query, **paper_data)
            logger.info(f"Upserted paper: {paper_data.get('id')}")

    def add_citation(self, source_id, target_id, context=None):
        """
        Adds a CITES relationship.
        """
        query = """
        MATCH (source:Paper {id: $source_id})
        MATCH (target:Paper {id: $target_id})
        MERGE (source)-[r:CITES]->(target)
        SET r.context = $context
        RETURN type(r)
        """
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, context=context)
            logger.info(f"Added citation: {source_id} -> {target_id}")

    def add_semantic_relation(self, source_id, target_id, relation_data):
        """
        Adds a SEMANTIC_RELATION relationship.
        relation_data: dict containing 'relation_type', 'confidence', 'reasoning', 'source'
        """
        query = """
        MATCH (source:Paper {id: $source_id})
        MATCH (target:Paper {id: $target_id})
        MERGE (source)-[r:SEMANTIC_RELATION {source: $source}]->(target)
        SET r.relation_type = $relation_type,
            r.confidence = $confidence,
            r.reasoning = $reasoning
        RETURN type(r)
        """
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, **relation_data)
            logger.info(f"Added semantic relation: {source_id} -[{relation_data.get('relation_type')}]-> {target_id}")

def main():
    # Configuration
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    loader = GraphLoader(uri, user, password)
    
    try:
        loader.setup_schema()
        
        # --- Demo Data Loading ---
        
        # 1. Create Paper A (Foundation Models)
        paper_a = {
            "id": "2108.07258",
            "title": "On the Opportunities and Risks of Foundation Models",
            "year": 2021,
            "venue": "arXiv",
            "paper_type": "Survey",
            "problem_statement": "Investigates an emerging paradigm for building AI systems based on foundation models.",
            "core_method": "Self-supervision at scale on broad data.",
            "key_findings": "Foundation models acquire surprising emergent capabilities but pose risks.",
            "embedding_problem": [0.1] * 384, # Dummy vector
            "embedding_method": [0.2] * 384
        }
        loader.add_paper(paper_a)
        
        # 2. Create Paper B (TaxoAdapt)
        paper_b = {
            "id": "2506.10737",
            "title": "TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction",
            "year": 2025,
            "venue": "ACL",
            "paper_type": "Methodology",
            "problem_statement": "Existing taxonomy construction methods fail to adapt to evolving corpora.",
            "core_method": "Synergizes LLM general knowledge with corpus-specific knowledge using a DAG structure.",
            "key_findings": "TaxoAdapt creates richer and more relevant taxonomies than baselines.",
            "embedding_problem": [0.3] * 384, # Dummy vector
            "embedding_method": [0.4] * 384
        }
        loader.add_paper(paper_b)
        
        # 3. Add Semantic Relation (Result from View T / View L)
        # Based on our previous analysis, they were "Likely Unrelated", but let's add a dummy relation for demo
        relation = {
            "relation_type": "Unrelated",
            "confidence": 0.85,
            "reasoning": "Paper A is a broad survey on foundation models, while Paper B is a specific method for taxonomy construction. No direct dependency.",
            "source": "View T + Heuristic"
        }
        loader.add_semantic_relation("2108.07258", "2506.10737", relation)
        
        print("Data loading complete.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        loader.close()

if __name__ == "__main__":
    main()
