import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import matplotlib.gridspec as gridspec # pyright: ignore[reportMissingImports]
from PIL import Image  
import json
from pathlib import Path
import torch
from typing import List, Literal, Optional


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity for already-normalized vectors."""
    return float(torch.dot(a, b))

def fuse_embeddings(
    v_img: Optional[torch.Tensor],
    v_txt: Optional[torch.Tensor],
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Weighted fusion for image+text query.
    alpha=1.0 => pure image, alpha=0.0 => pure text
    """
    if v_img is None and v_txt is None:
        raise ValueError("Provide at least one of v_img or v_txt")

    if v_img is None:
        v = v_txt
    elif v_txt is None:
        v = v_img
    else:
        v = alpha * v_img + (1.0 - alpha) * v_txt

    # normalize again after mixing
    v = v / v.norm()
    return v
    
def read_image_descriptions(image_descriptions_file: str) -> list[dict]:
    """
    Read image descriptions from a JSON file.
    """
    json_path = Path(image_descriptions_file)
    if not json_path.exists():
        print(f"Error: File {image_descriptions_file} does not exist")
        return []

    with open(json_path, 'r', encoding='utf-8') as jsonfile:
        image_descriptions = json.load(jsonfile)
    return image_descriptions

def save_embeddings(embeddings: list[dict], output_file: str) -> None:
    """
    Save embeddings to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(embeddings, jsonfile, indent=2, ensure_ascii=False)
    print(f"Saved embeddings to {output_file}")

def load_embeddings(input_file: str) -> list[dict]:
    """
    Load embeddings from a JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as jsonfile:
        embeddings = json.load(jsonfile)
    return embeddings

def rank_images(embeddings: list[dict], txt_query_embeddings: Optional[torch.Tensor] = None, image_query_embeddings: Optional[torch.Tensor] = None, embedding_type: Optional[Literal["fuse", "image", "text"]] = 'fuse') -> list[dict]:
    if txt_query_embeddings is None and image_query_embeddings is None:
        raise ValueError("Need text or image query")

    v_txt = None
    v_img = None
    if txt_query_embeddings is not None:
        v_txt = txt_query_embeddings
    if image_query_embeddings is not None:
        v_img = image_query_embeddings
    
    if embedding_type == 'fuse':
        v = fuse_embeddings(v_img, v_txt)
    elif embedding_type == 'image':
        v = v_img
    elif embedding_type == 'text':
        v = v_txt
    else:
        raise ValueError(f"Invalid embedding type: {embedding_type}")

    scores = []
    for embedding in embeddings:
        # Convert embedding list to tensor if it's a list
        fuse_emb = embedding[f'{embedding_type}_embedding']
        if isinstance(fuse_emb, list):
            fuse_emb = torch.tensor(fuse_emb)
        score = cosine(v, fuse_emb)
        scores.append({
            'filename': embedding['filename'],
            'description': embedding['description'],
            'score': score
        })
    return sorted(scores, key=lambda x: x['score'], reverse=True)

def show_test_results(test_results: list[dict], image_folder: str = 'images') -> None:
    """
    Visualize test results interactively with predictions on left and ground truth on right.
    Shows number of predicted images = ground_truth_count + 1
    User presses Enter in the visualization window to go to next result.
    """    
    plt.ion()  # Turn on interactive mode
    
    for idx, result in enumerate(test_results, 1):
        query = result['query']
        ground_truth = result['ground_truth']
        ranked_results = result['ranked_results']
        
        # Number of predictions to show = ground truth count + 1
        num_predictions = len(ground_truth) + 1
        num_ground_truth = len(ground_truth)
        
        # Get top N predicted images
        predicted_images = ranked_results[:num_predictions]
        
        # Maximum figure dimensions (in inches) - reasonable size to fit on most screens
        max_width = 16  # Maximum width in inches
        max_height = 14  # Maximum height in inches (fits most screens with some margin)
        base_width = 16  # Base width per image
        image_height_per_row = 2  # Height per image row in inches
        
        # Calculate desired size based on number of images
        desired_width = base_width
        desired_height = num_predictions * image_height_per_row
        
        # Scale down proportionally if figure would be too large for screen
        # Use the more restrictive scale (smaller value) to ensure it fits both dimensions
        width_ratio = max_width / desired_width
        height_ratio = max_height / desired_height
        scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't scale up, only down
        
        # Apply scaling
        fig_width = desired_width * scale_factor
        fig_height = desired_height * scale_factor
        
        # Create figure with scaled size
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(f'Test {idx}/{len(test_results)}: Query: "{query}"\nPress Enter to continue...', 
                    fontsize=16, fontweight='bold')
        
        # Create grid: 2 columns, max(num_predictions, num_ground_truth) rows
        gs = gridspec.GridSpec(num_predictions, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # Left column: Predictions
        for i, pred in enumerate(predicted_images):
            ax = fig.add_subplot(gs[i, 0])
            try:
                img_path = f"{image_folder}/{pred['filename']}"
                if Path(img_path).exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(f'Pred #{i+1}: {pred["filename"]}\nScore: {pred["score"]:.4f}', 
                               fontsize=10, pad=5)
                else:
                    ax.text(0.5, 0.5, f'Image not found:\n{pred["filename"]}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Pred #{i+1}: {pred["filename"]}', fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Pred #{i+1}: {pred["filename"]}', fontsize=10)
            ax.axis('off')
        
        # Add column header for predictions
        fig.text(0.25, 0.98, f'PREDICTED (Top {num_predictions})', 
                ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure)
        
        # Right column: Ground Truth
        for i, gt_filename in enumerate(ground_truth):
            ax = fig.add_subplot(gs[i, 1])
            try:
                img_path = f"{image_folder}/{gt_filename}"
                if Path(img_path).exists():
                    img = Image.open(img_path)
                    ax.imshow(img)
                    # Check if this ground truth is in predictions
                    in_predictions = any(p['filename'] == gt_filename for p in predicted_images)
                    color = 'green' if in_predictions else 'red'
                    ax.set_title(f'GT #{i+1}: {gt_filename}', 
                               fontsize=10, pad=5, color=color, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, f'Image not found:\n{gt_filename}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'GT #{i+1}: {gt_filename}', fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading:\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'GT #{i+1}: {gt_filename}', fontsize=10)
            ax.axis('off')
        
        # Add column header for ground truth
        fig.text(0.75, 0.98, f'GROUND TRUTH ({num_ground_truth})', 
                ha='center', fontsize=12, fontweight='bold', transform=fig.transFigure)
        
        # Make figure active and bring to front
        plt.show(block=False)
        try:
            # Try to bring window to front (works on some backends)
            fig.canvas.manager.window.raise_()
        except AttributeError:
            # On macOS or some backends, window attribute might not exist
            pass
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Wait for Enter key press in the figure window
        proceed = False
        def on_key_press(event):
            nonlocal proceed
            if event.key == 'enter':
                proceed = True
        
        cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Wait until Enter is pressed
        while not proceed:
            plt.pause(0.1)
        
        # Disconnect the event handler
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
    
    plt.ioff()  # Turn off interactive mode
    print(f"\n✓ Finished displaying all {len(test_results)} test results!")