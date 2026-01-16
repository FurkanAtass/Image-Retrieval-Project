import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from scripts.utils import read_image_descriptions, save_embeddings, load_embeddings, show_test_results, rank_images
from scripts.generate_descriptions import generate_image_descriptions

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
client = OpenAI(api_key=api_key)

# OpenAI embedding model
TEXT_EMBEDDING_MODEL = "text-embedding-3-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)

@torch.no_grad()
def embed_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    v = model.get_image_features(**inputs)  # [1, D]
    v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).cpu()

def embed_text(text: str) -> np.ndarray:
    """
    Get text embedding using OpenAI API.
    Returns normalized embedding vector as numpy array.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=TEXT_EMBEDDING_MODEL
        )
        embedding = np.array(response.data[0].embedding)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        raise Exception(f"Error creating embedding: {str(e)}")

def compute_embeddings(image_folder: str, image_descriptions_file: str, embeddings_file: str) -> list[dict]:
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
            image_embedding = embed_image(f"{image_folder}/{image_description['filename']}")
            embeddings.append({
                'filename': image_description['filename'],
                'description': image_description['description'],
                'text_embedding': text_embedding.tolist(),
                'image_embedding': image_embedding.tolist(),
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


def generate_test_results(embeddings: list[dict], test_cases_file: str, query_image_folder: str) -> list[dict]:
    """
    Generate test results for a list of test cases.
    """
    test_cases = json.load(open(test_cases_file, 'r', encoding='utf-8'))
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"Processing test case {idx}/{len(test_cases)}: '{test_case['query'] if test_case.get('query', None) is not None else test_case['query_image']}'...")

        img_query_embeddings = embed_image(f"{query_image_folder}/{test_case['query_image']}") if test_case.get('query_image', None) is not None else None
  
        txt_query_embeddings = embed_text(test_case['query']) if test_case.get('query', None) is not None else None

        rank_results = rank_images(embeddings, txt_query_embeddings, img_query_embeddings)
        results.append({
            'query': test_case.get('query', None),
            'query_image': test_case.get('query_image', None),
            'ground_truth': test_case['top_results'],
            'ranked_results': rank_results
        })
    
    return results

def main():
    """
    Main function to compute embeddings and rank images using OpenAI text embeddings.
    """
    image_folder = 'dataset/images'
    image_descriptions_file = 'dataset/image_descriptions.json'
    embeddings_file = 'precomputed_embeddings/openai_embeddings.json'
    test_cases_file = 'dataset/test_cases.json'
    query_image_folder = 'dataset/query_images'

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
        embeddings = compute_embeddings(image_folder, image_descriptions_file, embeddings_file)
    else:
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = load_embeddings(embeddings_file)
        print(f"Loaded {len(embeddings)} valid embeddings")

    print("\nGenerating test results...")
    test_results = generate_test_results(embeddings, test_cases_file, query_image_folder)
    
    print("\nDisplaying results...")
    show_test_results(test_results, image_folder, query_image_folder)

if __name__ == "__main__":
    main()
