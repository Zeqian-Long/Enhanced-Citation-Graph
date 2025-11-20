import os
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv  # <--- IMPORT THIS

# --- LOAD DOTENV ---
# This finds the .env file and loads it into os.environ
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAG:
    def __init__(self, uri, user, password, embedding_model_name='all-MiniLM-L6-v2'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def close(self):
        self.driver.close()

    def get_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    def retrieve_anchor_nodes(self, query_text, k=3):
        """
        Retrieves the top-k most relevant papers using vector similarity on the problem statement.
        """
        query_embedding = self.get_embedding(query_text)
        
        cypher_query = """
        CALL db.index.vector.queryNodes('paper_problem_embedding', $k, $embedding)
        YIELD node, score
        RETURN node.id AS id, node.title AS title, node.problem_statement AS problem, score
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, k=k, embedding=query_embedding)
            anchors = [record.data() for record in result]
            
        logger.info(f"Retrieved {len(anchors)} anchor nodes.")
        return anchors

    def expand_subgraph(self, anchor_ids):
        """
        Retrieves the 1-hop neighborhood of the anchor nodes, focusing on semantic relations.
        """
        cypher_query = """
        MATCH (origin:Paper)-[r]-(neighbor:Paper)
        WHERE origin.id IN $anchor_ids
        RETURN 
            origin.title AS origin_title, 
            type(r) AS edge_type,
            r.relation_type AS semantic_relation,
            r.reasoning AS reasoning,
            neighbor.title AS neighbor_title, 
            neighbor.problem_statement AS neighbor_problem,
            neighbor.core_method AS neighbor_method
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, anchor_ids=anchor_ids)
            subgraph = [record.data() for record in result]
            
        logger.info(f"Expanded subgraph contains {len(subgraph)} edges.")
        return subgraph

    def construct_prompt(self, user_query, anchors, subgraph):
        """
        Constructs the prompt for the LLM using the retrieved graph context.
        """
        context_str = "### Retrieved Papers (Anchors):\n"
        for p in anchors:
            context_str += f"- **{p['title']}** (Score: {p['score']:.2f})\n"
            context_str += f"  Problem: {p['problem']}\n\n"
            
        context_str += "### Related Work (Graph Connections):\n"
        for edge in subgraph:
            relation = edge['semantic_relation'] if edge['semantic_relation'] else edge['edge_type']
            context_str += f"- **{edge['origin_title']}** --[{relation}]--> **{edge['neighbor_title']}**\n"
            if edge['reasoning']:
                context_str += f"  Reasoning: {edge['reasoning']}\n"
            context_str += f"  Neighbor Method: {edge['neighbor_method']}\n\n"
            
        prompt = f"""
You are an expert scientific assistant. Answer the user's question using the provided context from the Citation Graph.
Use the semantic relations (e.g., Extend, Contrast, Support) to explain *how* the papers are related, not just *that* they are related.

User Query: "{user_query}"

{context_str}

Answer:
"""
        return prompt

    def generate_answer(self, prompt):
        """
        Simulates calling an LLM. In production, replace this with OpenAI/Anthropic API calls.
        """
        print("\n" + "="*50)
        print("GENERATED PROMPT FOR LLM:")
        print("="*50)
        print(prompt)
        print("="*50)
        
        # Simulated response
        return "(This is where the LLM would generate a synthesized answer based on the prompt above.)"

    def run_pipeline(self, user_query):
        print(f"Processing Query: '{user_query}'")
        
        # 1. Retrieve Anchors
        anchors = self.retrieve_anchor_nodes(user_query)
        if not anchors:
            print("No relevant papers found.")
            return

        anchor_ids = [a['id'] for a in anchors]
        
        # 2. Expand Graph
        subgraph = self.expand_subgraph(anchor_ids)
        
        # 3. Generate Answer
        prompt = self.construct_prompt(user_query, anchors, subgraph)
        answer = self.generate_answer(prompt)
        
        return answer

def main():
    # Configuration
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    rag = GraphRAG(uri, user, password)
    
    try:
        # Demo Query
        query = "How do recent approaches improve upon foundation models?"
        
        # Note: This will likely return empty results if the Neo4j DB is empty or not running.
        # To see it work, ensure graph_loader.py has been run against a live DB.
        rag.run_pipeline(query)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print("\nNOTE: Ensure Neo4j is running and populated (run graph_loader.py first).")
    finally:
        rag.close()

if __name__ == "__main__":
    main()
