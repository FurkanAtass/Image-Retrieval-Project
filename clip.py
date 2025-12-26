import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional
import json
from pathlib import Path
from utils import read_image_descriptions, save_embeddings, load_embeddings, cosine, show_test_results, fuse_embeddings, rank_images
from generate_descriptions import generate_image_descriptions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(model_id)

@torch.no_grad()
def embed_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    v = model.get_image_features(**inputs)  # [1, D]
    v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).cpu()

@torch.no_grad()
def embed_text(text: str) -> torch.Tensor:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
    v = model.get_text_features(**inputs)  # [1, D]
    v = v / v.norm(dim=-1, keepdim=True)
    return v.squeeze(0).cpu()



def compute_embeddings(image_folder: str, image_descriptions_file: str) -> list[dict]:
    """
    Compute embeddings for all images in the image folder and save to a list of dictionaries.
    """
    image_descriptions = read_image_descriptions(image_descriptions_file)
    embeddings = []

    for image_description in image_descriptions:
        v_img = embed_image(f"{image_folder}/{image_description['filename']}")
        v_txt = embed_text(image_description['description'])
        v = fuse_embeddings(v_img, v_txt)
        embeddings.append({
            'filename': image_description['filename'],
            'description': image_description['description'],
            'image_embedding': v_img.tolist(),
            'text_embedding': v_txt.tolist(),
            'fuse_embedding': v.tolist(),
        })
    save_embeddings(embeddings, "clip_embeddings.json")
    return embeddings



def generate_test_results(embeddings: list[dict], test_cases_file: str) -> list[dict]:
    """
    Generate test results for a list of test cases.
    """
    test_cases = json.load(open(test_cases_file, 'r', encoding='utf-8'))
    results = []
    for test_case in test_cases:
        txt_query_embeddings = embed_text(test_case['query'])
        rank_results = rank_images(embeddings, txt_query_embeddings)
        results.append({
            'query': test_case['query'],
            'ground_truth': test_case['top_results'],
            'ranked_results': rank_results
        })
    return results

def main():
    """
    Main function to compute embeddings and save to a JSON file.
    """
    image_folder = 'images'
    image_descriptions_file = 'image_descriptions.json'
    embeddings_file = 'clip_embeddings.json'
    test_cases_file = 'test_cases.json'

    if not Path(image_folder).exists():
        print(f"Error: Directory {image_folder} does not exist")
        return
    if not Path(image_descriptions_file).exists():
        print(f"Image Descriptions File {image_descriptions_file} does not exist")
        generate_image_descriptions(image_folder, image_descriptions_file)
    if not Path(embeddings_file).exists():
        print(f"Embeddings File {embeddings_file} does not exist")
        embeddings = compute_embeddings(image_folder, image_descriptions_file)
    else:
        embeddings = load_embeddings(embeddings_file)

    test_results = generate_test_results(embeddings, test_cases_file)
    show_test_results(test_results, image_folder)

if __name__ == "__main__":
    main()