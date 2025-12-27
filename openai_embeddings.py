import os
import json
import numpy as np
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from utils import read_image_descriptions, save_embeddings, load_embeddings, show_test_results, rank_images
from generate_descriptions import generate_image_descriptions

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
client = OpenAI(api_key=api_key)

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> np.ndarray:
    """
    Get text embedding using OpenAI API.
    Returns normalized embedding vector as numpy array.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = np.array(response.data[0].embedding)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        raise Exception(f"Error creating embedding: {str(e)}")

def compute_embeddings(image_descriptions_file: str, embeddings_file: str) -> list[dict]:
    """
    Compute text embeddings for all image descriptions and save to a list of dictionaries.
    Only uses text descriptions, no image embeddings.
    """
    image_descriptions = read_image_descriptions(image_descriptions_file)
    embeddings = []

    print(f"Computing embeddings for {len(image_descriptions)} descriptions...")
    
    for idx, image_description in enumerate(image_descriptions, 1):
        print(f"[{idx}/{len(image_descriptions)}] Processing {image_description['filename']}...", end=' ', flush=True)
        
        try:
            # Only embed text description (no image embedding)
            text_embedding = embed_text(image_description['description'])
            
            embeddings.append({
                'filename': image_description['filename'],
                'description': image_description['description'],
                'text_embedding': text_embedding.tolist(),
            })
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Still add entry with error
            embeddings.append({
                'filename': image_description['filename'],
                'description': image_description['description'],
                'text_embedding': None,
            })
    
    save_embeddings(embeddings, embeddings_file)
    return embeddings


def generate_test_results(embeddings: list[dict], test_cases_file: str) -> list[dict]:
    """
    Generate test results for a list of test cases.
    """
    test_cases = json.load(open(test_cases_file, 'r', encoding='utf-8'))
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"Processing test case {idx}/{len(test_cases)}: '{test_case['query']}'...")
        txt_query_embeddings = embed_text(test_case['query'])
        rank_results = rank_images(embeddings, txt_query_embeddings, embedding_type='text')
        results.append({
            'query': test_case['query'],
            'ground_truth': test_case['top_results'],
            'ranked_results': rank_results
        })
    
    return results

def main():
    """
    Main function to compute embeddings and rank images using OpenAI text embeddings.
    """
    image_folder = 'images'
    image_descriptions_file = 'image_descriptions.json'
    embeddings_file = 'openai_embeddings.json'
    test_cases_file = 'test_cases.json'

    if not Path(image_folder).exists():
        print(f"Error: Directory {image_folder} does not exist")
        return
    
    if not Path(image_descriptions_file).exists():
        print(f"Image Descriptions File {image_descriptions_file} does not exist")
        print("Generating descriptions...")
        generate_image_descriptions(image_folder, image_descriptions_file)
    
    if not Path(embeddings_file).exists():
        print(f"Embeddings File {embeddings_file} does not exist")
        print("Computing embeddings...")
        embeddings = compute_embeddings(image_descriptions_file, embeddings_file)
    else:
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = load_embeddings(embeddings_file)
        # Filter out embeddings with None (errors)
        embeddings = [e for e in embeddings if e.get('text_embedding') is not None]
        print(f"Loaded {len(embeddings)} valid embeddings")

    print("\nGenerating test results...")
    test_results = generate_test_results(embeddings, test_cases_file)
    
    print("\nDisplaying results...")
    show_test_results(test_results, image_folder)

if __name__ == "__main__":
    main()
