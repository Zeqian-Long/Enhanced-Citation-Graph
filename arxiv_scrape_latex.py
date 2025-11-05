#!/usr/bin/env python3

"""
A script to download and extract the LaTeX source of an arXiv paper.

Usage:
    python scrape_arxiv.py <url_or_id> [-d /path/to/download_dir]

Example:
    # Using a new paper ID
    python scrape_arxiv.py 2301.00001

    # Using an old paper ID
    python scrape_arxiv.py hep-th/0001001

    # Using a full URL
    python scrape_arxiv.py https://arxiv.org/abs/2401.12345v1

    # Specifying a download directory
    python scrape_arxiv.py 2401.00002 -d ./my_papers
"""

import requests
import re
import os
import tarfile
import io
import argparse
import sys

def scrape_arxiv_source(url_or_id: str, download_dir: str = 'arxiv_source'):
    """
    Downloads and extracts the LaTeX source for a given arXiv paper URL or ID.

    Args:
        url_or_id (str): The arXiv URL (abs or pdf) or paper ID.
        download_dir (str): The top-level directory to save source files.
    """
    print(f"Attempting to fetch source for: {url_or_id}")

    # Regex to find new (e.g., 2301.00001) or old (e.g., hep-th/0001001) arXiv IDs
    # This will find the ID even if it's part of a full URL.
    id_match = re.search(
        r'([\w-]+/\d{7}(?:v\d+)?|\d{4}\.\d{5}(?:v\d+)?)',
        url_or_id
    )

    if not id_match:
        print(f"Error: Could not parse a valid arXiv ID from '{url_or_id}'", file=sys.stderr)
        return
    
    arxiv_id = id_match.group(1)
    print(f"Found arXiv ID: {arxiv_id}")

    # Construct the source URL
    src_url = f'https://arxiv.org/src/{arxiv_id}'

    # Define a clean directory name for the paper (replaces '/' in old IDs)
    paper_dir_name = arxiv_id.replace('/', '_')
    paper_download_path = os.path.join(download_dir, paper_dir_name)

    # Create the directory
    os.makedirs(paper_download_path, exist_ok=True)

    try:
        # Fetch the source. A User-Agent is good practice.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(src_url, headers=headers, allow_redirects=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)

        content_type = response.headers.get('Content-Type', '')
        print(f"Server responded with Content-Type: {content_type}")

        if 'application/x-gzip' in content_type or 'application/x-tar' in content_type or 'application/gzip' in content_type:
            # It's a gzipped tarball
            print(f"Detected .tar.gz archive. Extracting to {paper_download_path}...")
            file_like_object = io.BytesIO(response.content)
            with tarfile.open(fileobj=file_like_object, mode='r:gz') as tar:
                tar.extractall(path=paper_download_path)
            print(f"Successfully extracted archive to: {paper_download_path}")

        elif 'application/x-tex' in content_type or 'text/plain' in content_type:
            # It's a single .tex file
            print("Detected single .tex file. Saving...")
            filename = f"{paper_dir_name}.tex"
            
            # Try to get a better filename from Content-Disposition
            content_disposition = response.headers.get('Content-Disposition')
            if content_disposition:
                filename_match = re.search(r'filename="(.+?)"', content_disposition)
                if filename_match:
                    filename = filename_match.group(1)

            filepath = os.path.join(paper_download_path, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved file to: {filepath}")

        else:
            # Unknown content type
            print(f"Warning: Unknown Content-Type '{content_type}'. Saving raw content.")
            filepath = os.path.join(paper_download_path, 'unknown_source_file')
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Raw content saved to: {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching source: {e}", file=sys.stderr)
    except tarfile.TarError as e:
        print(f"Error extracting tarball: {e}", file=sys.stderr)
    except IOError as e:
        print(f"Error writing file: {e}", file=sys.stderr)

def main():
    """Main function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Download and extract the LaTeX source of an arXiv paper.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'url_or_id',
        help='The arXiv URL (abs or pdf) or paper ID.'
    )
    parser.add_argument(
        '-d', '--dir',
        default='arxiv_source',
        help='Directory to save the source files (default: "arxiv_source").'
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    scrape_arxiv_source(args.url_or_id, args.dir)

if __name__ == '__main__':
    main()

