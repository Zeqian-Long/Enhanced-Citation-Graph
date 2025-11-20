import os
import re
import json

def read_tex_file(file_path, base_dir):
    """
    Reads a tex file and recursively resolves \input{} commands.
    """
    if not os.path.exists(file_path):
        # Try adding .tex extension
        if os.path.exists(file_path + ".tex"):
            file_path += ".tex"
        else:
            print(f"Warning: File not found: {file_path}")
            return ""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find \input{filename}
    # Note: filename might not have .tex extension
    input_pattern = re.compile(r'\\input\{([^}]+)\}')
    
    def replace_input(match):
        sub_file = match.group(1)
        sub_file_path = os.path.join(base_dir, sub_file)
        return read_tex_file(sub_file_path, base_dir)

    content = input_pattern.sub(replace_input, content)
    return content

def extract_section(full_text, section_name):
    """
    Extracts text belonging to a specific section (e.g., Introduction).
    This is a heuristic extraction based on \section{Name}.
    """
    # Normalize section name for regex
    # \section{Introduction} or \section{Introduction \label{...}}
    pattern = re.compile(r'\\section\{' + re.escape(section_name) + r'.*?\}(.*?)\\section\{', re.DOTALL | re.IGNORECASE)
    match = pattern.search(full_text)
    
    if match:
        return match.group(1).strip()
    
    # If it's the last section, it might not be followed by another \section
    pattern_last = re.compile(r'\\section\{' + re.escape(section_name) + r'.*?\}(.*)', re.DOTALL | re.IGNORECASE)
    match_last = pattern_last.search(full_text)
    
    if match_last:
        return match_last.group(1).strip()
        
    return None

def clean_latex(text):
    """
    Removes basic LaTeX commands for cleaner prompt input.
    """
    if not text: return ""
    # Remove comments
    text = re.sub(r'%.*', '', text)
    # Remove \cite{...}
    text = re.sub(r'\\cite\{.*?\}', '[CITATION]', text)
    text = re.sub(r'\\citep\{.*?\}', '[CITATION]', text)
    text = re.sub(r'\\citet\{.*?\}', '[CITATION]', text)
    # Remove \ref{...}
    text = re.sub(r'\\ref\{.*?\}', '[REF]', text)
    # Remove other commands like \textbf{}, \textit{} but keep content
    text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', text)
    # Remove simple commands like \noindent
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    base_dir = "arxiv_source/2506.10737"
    root_file = "acl_latex.tex"
    full_path = os.path.join(base_dir, root_file)
    
    print(f"Processing {full_path}...")
    full_text = read_tex_file(full_path, base_dir)
    
    # print(f"Full text length: {len(full_text)}")
    
    # Extract Sections
    intro_text = extract_section(full_text, "Introduction")
    method_text = extract_section(full_text, "Methodology") # In this paper it is called Methodology
    if not method_text:
         method_text = extract_section(full_text, "Method")
         
    experiments_text = extract_section(full_text, "Experiments")
    
    # Load Prompts
    with open("prompts.md", "r") as f:
        prompts_content = f.read()
        
    # Parse prompts from markdown (simple split)
    # This is just for demo, in production use a proper parser or separate files
    
    print("\n" + "="*50)
    print("DEMO: Generating Prompt for Problem Extraction")
    print("="*50)
    
    print("SYSTEM PROMPT:")
    print("""You are a senior research scientist analyzing a paper's introduction. Extract the core problem and contribution.

Output a JSON object with the following structure:
{
  "problem_statement": "string (Concise description of the specific problem the paper aims to solve)",
  "motivation": "string (Why is this problem important? What are the limitations of existing solutions?)",
  "contribution_type": "string (Select one: 'New Algorithm', 'New Dataset', 'Performance Improvement', 'Theoretical Proof', 'System Architecture', 'Empirical Study')",
  "core_contribution": "string (A 1-2 sentence summary of the main contribution)"
}""")
    
    print("\nUSER INPUT:")
    print(f"Paper Introduction:\n{clean_latex(intro_text)[:2000]}...") # Truncate for display

    print("\n" + "="*50)
    print("DEMO: Generating Prompt for Method Extraction")
    print("="*50)
    
    print("SYSTEM PROMPT:")
    print("""You are a technical expert in this field. Summarize the methodology described in the text. Focus on the *how*, not the *why*.

Output a JSON object with the following structure:
{
  "method_name": "string (The name given to the method, or 'Proposed Method' if unnamed)",
  "core_approach": "string (A 2-3 sentence technical summary of the framework, architecture, or algorithm)",
  "key_components": ["string", "string"] (List of main modules or steps)
}""")
    
    print("\nUSER INPUT:")
    print(f"Paper Method Section:\n{clean_latex(method_text)[:2000]}...")

if __name__ == "__main__":
    main()
