import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from extract_relation import read_tex_file, extract_section, clean_latex

def get_embeddings(text_list, model):
    """
    Generates embeddings for a list of texts using the provided model.
    """
    # Encode texts
    embeddings = model.encode(text_list)
    return embeddings

def compute_view_t_metrics(paper_a_data, paper_b_data, model_name='allenai/specter'):
    """
    Computes View T (Textual Similarity) metrics between two papers.
    """
    print(f"Loading model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to 'all-MiniLM-L6-v2'")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare texts
    texts = [
        paper_a_data['problem'],
        paper_a_data['method'],
        paper_b_data['problem'],
        paper_b_data['method']
    ]
    
    print("Generating embeddings...")
    embeddings = get_embeddings(texts, model)
    
    # Unpack embeddings
    emb_prob_a = embeddings[0].reshape(1, -1)
    emb_meth_a = embeddings[1].reshape(1, -1)
    emb_prob_b = embeddings[2].reshape(1, -1)
    emb_meth_b = embeddings[3].reshape(1, -1)
    
    # Compute Similarities
    sim_prob = cosine_similarity(emb_prob_a, emb_prob_b)[0][0]
    sim_meth = cosine_similarity(emb_meth_a, emb_meth_b)[0][0]
    
    # Cross similarities (optional, but useful for "Method Reuse" detection)
    # e.g. Does Method A match Problem B? (Maybe not directly comparable, but Method A description vs Method B description is key)
    
    return sim_prob, sim_meth

def analyze_similarity(sim_prob, sim_meth, threshold_high=0.75, threshold_low=0.5):
    """
    Heuristic analysis based on similarity scores.
    """
    print("\n--- View T Analysis ---")
    print(f"Problem Similarity: {sim_prob:.4f}")
    print(f"Method Similarity:  {sim_meth:.4f}")
    
    candidates = []
    
    if sim_prob > threshold_high:
        if sim_meth > threshold_high:
            candidates.append("Potential Duplicate / Replication / Incremental Improvement")
        elif sim_meth < threshold_low:
            candidates.append("Alternative Approach (Same Problem, Different Method)")
        else:
            candidates.append("Related Work (Same Problem, Moderate Method Overlap)")
            
    elif sim_prob < threshold_low:
        if sim_meth > threshold_high:
            candidates.append("Method Reuse / Transfer Learning (Different Problem, Same Method)")
        else:
            candidates.append("Likely Unrelated (or weak connection)")
            
    else:
        candidates.append("Loosely Related")
        
    print(f"Heuristic Classification: {', '.join(candidates)}")
    return candidates

def main():
    # Define paths (Reusing the setup from extract_relation.py)
    paper_a_dir = "arxiv_source/2108.07258" # Foundation Models
    paper_a_root = "main.tex"
    
    paper_b_dir = "arxiv_source/2506.10737" # TaxoAdapt
    paper_b_root = "acl_latex.tex"
    
    print("Reading Paper A...")
    text_a = read_tex_file(os.path.join(paper_a_dir, paper_a_root), paper_a_dir)
    
    print("Reading Paper B...")
    text_b = read_tex_file(os.path.join(paper_b_dir, paper_b_root), paper_b_dir)
    
    # Extract Sections
    # Note: For Foundation Models, we use 'Capabilities' as a proxy for Method/Approach for this demo
    prob_a = clean_latex(extract_section(text_a, "Introduction"))
    method_a = clean_latex(extract_section(text_a, "Capabilities")) 
    
    prob_b = clean_latex(extract_section(text_b, "Introduction"))
    method_b = clean_latex(extract_section(text_b, "Methodology"))
    
    if not prob_a or not method_a or not prob_b or not method_b:
        print("Error: Could not extract all sections. Please check the extraction logic.")
        # Fallback for demo if extraction fails (e.g. section names differ)
        if not method_a: method_a = "Foundation models are trained on broad data using self-supervision at scale."
        if not method_b: method_b = "TaxoAdapt aligns LLM taxonomy generation to a specific corpus using a DAG structure."
    
    paper_a_data = {'problem': prob_a, 'method': method_a}
    paper_b_data = {'problem': prob_b, 'method': method_b}
    
    # Compute Metrics
    # Using 'all-MiniLM-L6-v2' for speed/demo purposes, but 'allenai/specter' is better for papers
    sim_prob, sim_meth = compute_view_t_metrics(paper_a_data, paper_b_data, model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Analyze
    analyze_similarity(sim_prob, sim_meth)

if __name__ == "__main__":
    main()
