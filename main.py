import os
import logging
import json
from dotenv import load_dotenv

# Import components
from extract_node import read_tex_file, extract_section, clean_latex
from compute_similarity import compute_view_t_metrics, analyze_similarity
from graph_loader import GraphLoader
from graph_rag import GraphRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def step_1_extraction(paper_dirs):
    """
    Phase 1: Extract structured data from LaTeX source.
    In a real system, this would call an LLM API. 
    Here, we extract raw text and simulate the structured output.
    """
    logger.info("--- Phase 1: Node Extraction ---")
    extracted_papers = []

    for pid, pdir in paper_dirs.items():
        logger.info(f"Processing paper {pid}...")
        
        # 1. Read LaTeX
        # Heuristic: find the main tex file
        tex_files = [f for f in os.listdir(pdir) if f.endswith('.tex')]
        if not tex_files:
            logger.warning(f"No .tex file found in {pdir}")
            continue
            
        # Prefer 'main.tex' or 'acl_latex.tex' or the largest file
        main_tex = next((f for f in tex_files if 'main' in f), tex_files[0])
        full_text = read_tex_file(os.path.join(pdir, main_tex), pdir)
        
        # 2. Extract Sections
        intro = clean_latex(extract_section(full_text, "Introduction") or "")
        method = clean_latex(extract_section(full_text, "Methodology") or extract_section(full_text, "Capabilities") or "")
        
        # 3. Simulate LLM Output (Structured Node)
        # In production, you would send 'intro' and 'method' to the prompts in prompts.md
        paper_node = {
            "id": pid,
            "title": f"Simulated Title for {pid}", # In real app, extract from metadata
            "year": 2025, # Placeholder
            "problem_statement": intro[:500] + "...", # Truncated for demo
            "core_method": method[:500] + "...",
            "full_problem_text": intro, # Keep full text for embedding
            "full_method_text": method
        }
        extracted_papers.append(paper_node)
        logger.info(f"Extracted node for {pid}")

    return extracted_papers

def step_2_relation_discovery(papers):
    """
    Phase 2: Discover relations between papers.
    """
    logger.info("\n--- Phase 2: Relation Discovery ---")
    relations = []
    
    # Compare every pair (O(N^2)) - For demo, we only have 2 papers
    if len(papers) < 2:
        logger.warning("Not enough papers to compute relations.")
        return relations

    p1 = papers[0]
    p2 = papers[1]
    
    logger.info(f"Comparing {p1['id']} vs {p2['id']}...")
    
    # View T: Textual Similarity
    data_a = {'problem': p1['full_problem_text'], 'method': p1['full_method_text']}
    data_b = {'problem': p2['full_problem_text'], 'method': p2['full_method_text']}
    
    # Note: This requires sentence-transformers installed
    try:
        sim_prob, sim_meth = compute_view_t_metrics(data_a, data_b, model_name='all-MiniLM-L6-v2')
        candidates = analyze_similarity(sim_prob, sim_meth)
        
        # Create a relation object
        relation = {
            "source_id": p1['id'],
            "target_id": p2['id'],
            "relation_type": candidates[0].split(" ")[0] if candidates else "Unrelated", # Simple heuristic
            "confidence": (sim_prob + sim_meth) / 2,
            "reasoning": f"Computed via View T. Problem Sim: {sim_prob:.2f}, Method Sim: {sim_meth:.2f}",
            "source": "View T"
        }
        relations.append(relation)
        
    except Exception as e:
        logger.error(f"Failed to compute similarity: {e}")

    return relations

def step_3_storage(papers, relations):
    """
    Phase 3: Load data into Neo4j.
    """
    logger.info("\n--- Phase 3: Graph Storage ---")
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        loader = GraphLoader(uri, user, password)
        loader.setup_schema()
        
        # Load Papers
        for p in papers:
            # Generate dummy embeddings if not present (GraphLoader expects them)
            # In real app, compute them here or in Step 2
            p['embedding_problem'] = [0.1] * 384 
            p['embedding_method'] = [0.1] * 384
            
            # Remove full text fields before loading to DB to save space/cleanliness if needed
            # or keep them. GraphLoader expects specific keys.
            db_node = {
                "id": p['id'],
                "title": p['title'],
                "year": p['year'],
                "venue": "arXiv",
                "paper_type": "Unknown",
                "problem_statement": p['problem_statement'],
                "core_method": p['core_method'],
                "key_findings": "Simulated findings",
                "embedding_problem": p['embedding_problem'],
                "embedding_method": p['embedding_method']
            }
            loader.add_paper(db_node)
            
        # Load Relations
        for r in relations:
            loader.add_semantic_relation(
                r['source_id'], 
                r['target_id'], 
                {
                    "relation_type": r['relation_type'],
                    "confidence": float(r['confidence']),
                    "reasoning": r['reasoning'],
                    "source": r['source']
                }
            )
            
        loader.close()
        logger.info("Data successfully loaded into Neo4j.")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j Connection Failed: {e}")
        logger.warning("Skipping DB loading. Ensure Neo4j is running.")
        return False

def step_4_reasoning():
    """
    Phase 4: Run GraphRAG.
    """
    logger.info("\n--- Phase 4: Reasoning Engine ---")
    
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        rag = GraphRAG(uri, user, password)
        query = "How do these papers relate to each other?"
        
        logger.info(f"Running Query: {query}")
        # This will fail if DB is empty/unreachable, handled by try/except
        rag.run_pipeline(query)
        rag.close()
        
    except Exception as e:
        logger.error(f"GraphRAG Failed: {e}")

def main():
    load_dotenv()
    
    # Define input data
    base_dir = "arxiv_source"
    paper_dirs = {
        "2108.07258": os.path.join(base_dir, "2108.07258"),
        "2506.10737": os.path.join(base_dir, "2506.10737")
    }
    
    # Execute Pipeline
    papers = step_1_extraction(paper_dirs)
    relations = step_2_relation_discovery(papers)
    
    db_success = step_3_storage(papers, relations)
    
    if db_success:
        step_4_reasoning()
    else:
        logger.info("Skipping Phase 4 due to DB issues.")

if __name__ == "__main__":
    main()
