import os
import re
import json

# --- Helper Functions (Copied from extract_node.py for standalone execution) ---

def read_tex_file(file_path, base_dir):
    """
    Reads a tex file and recursively resolves \input{} commands.
    """
    # Check if file exists or needs .tex extension
    if os.path.exists(file_path) and os.path.isfile(file_path):
        pass # It's a valid file
    elif os.path.exists(file_path + ".tex"):
        file_path += ".tex"
    else:
        print(f"Warning: File not found: {file_path}")
        return ""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    input_pattern = re.compile(r'\\input(?:\{([^}]+)\}|\s+([^\s}]+))')
    
    def replace_input(match):
        sub_file = match.group(1) or match.group(2)
        sub_file_path = os.path.join(base_dir, sub_file)
        return read_tex_file(sub_file_path, base_dir)

    content = input_pattern.sub(replace_input, content)
    return content

def extract_section(full_text, section_name):
    """
    Extracts text belonging to a specific section.
    """
    pattern = re.compile(r'\\section\{' + re.escape(section_name) + r'.*?\}(.*?)\\section\{', re.DOTALL | re.IGNORECASE)
    match = pattern.search(full_text)
    
    if match:
        return match.group(1).strip()
    
    pattern_last = re.compile(r'\\section\{' + re.escape(section_name) + r'.*?\}(.*)', re.DOTALL | re.IGNORECASE)
    match_last = pattern_last.search(full_text)
    
    if match_last:
        return match_last.group(1).strip()
        
    return None

def clean_latex(text):
    if not text: return ""
    text = re.sub(r'%.*', '', text)
    text = re.sub(r'\\cite\{.*?\}', '[CITATION]', text)
    text = re.sub(r'\\citep\{.*?\}', '[CITATION]', text)
    text = re.sub(r'\\citet\{.*?\}', '[CITATION]', text)
    text = re.sub(r'\\ref\{.*?\}', '[REF]', text)
    text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Logic for Relation Extraction ---

def main():
    # Define paths for two papers
    paper_a_dir = "arxiv_source/2108.07258"
    paper_a_root = "main.tex" # Assuming main.tex for the Foundation Models paper
    
    paper_b_dir = "arxiv_source/2506.10737"
    paper_b_root = "acl_latex.tex"
    
    print("Loading Paper A (Foundation Models)...")
    text_a = read_tex_file(os.path.join(paper_a_dir, paper_a_root), paper_a_dir)
    
    print("Loading Paper B (TaxoAdapt)...")
    text_b = read_tex_file(os.path.join(paper_b_dir, paper_b_root), paper_b_dir)
    
    # Extract relevant sections (Simulating the output of Phase 1)
    # In a real pipeline, we would use the JSON output from the Node Extraction step.
    # Here we just grab the raw text as a proxy.
    
    prob_a = clean_latex(extract_section(text_a, "Introduction"))
    method_a = clean_latex(extract_section(text_a, "Capabilities")) # Using Capabilities as proxy for method in this survey
    
    prob_b = clean_latex(extract_section(text_b, "Introduction"))
    method_b = clean_latex(extract_section(text_b, "Methodology"))
    
    # Construct the Prompt
    
    system_prompt = """You are a senior scientific editor tasked with determining the relationship between two research papers.
You will be provided with the structured summaries of two papers: Paper A (the earlier or reference paper) and Paper B (the later or candidate paper).

Analyze their relationship based on the following criteria:
1. **Problem Overlap:** Do they solve the same problem?
2. **Method Similarity:** Do they use similar techniques?
3. **Result Consistency:** Do their results agree or disagree?

Classify the relationship into ONE of the following categories:
- **Extend:** Paper B builds directly upon Paper A.
- **Support:** Paper B provides evidence that confirms Paper A's findings.
- **Contrast:** Paper B refutes, critiques, or provides counter-evidence to Paper A.
- **Alternative Approach:** Paper B solves the same problem as Paper A but uses a significantly different method.
- **Method Reuse:** Paper B applies the method from Paper A to a new problem or domain.
- **Background:** Paper A is merely cited as context or prior work by Paper B.
- **Unrelated:** The papers have no significant semantic connection.

Output a JSON object with the following structure:
{
  "relation_type": "string",
  "confidence": "string",
  "reasoning": "string"
}"""

    user_prompt = f"""
Paper A (Reference):
- Problem Statement: {prob_a[:500]}...
- Core Approach: {method_a[:500]}...

Paper B (Candidate):
- Problem Statement: {prob_b[:500]}...
- Core Approach: {method_b[:500]}...
"""

    print("\n" + "="*50)
    print("DEMO: Generating Pairwise Relation Prompt")
    print("="*50)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("\nUSER PROMPT:")
    print(user_prompt)

if __name__ == "__main__":
    main()
