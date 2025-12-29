"""
Example usage of vector databases for image retrieval.
Shows how to migrate from JSON to Chroma/Qdrant.
"""

from pathlib import Path
from utils import load_embeddings
from vector_db_chroma import ChromaVectorDB
# from vector_db_qdrant import QdrantVectorDB  # Uncomment to use Qdrant

def migrate_json_to_chroma(json_file: str, db_path: str = "./chroma_db"):
    """Migrate embeddings from JSON to Chroma."""
    print("Loading embeddings from JSON...")
    embeddings = load_embeddings(json_file)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Initialize Chroma
    print("\nInitializing Chroma database...")
    db = ChromaVectorDB(db_path=db_path)
    
    # Add embeddings
    print("\nAdding embeddings to Chroma...")
    db.add_embeddings(embeddings)
    
    # Get stats
    stats = db.get_stats()
    print(f"\n✓ Migration complete!")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    
    return db

def example_search(db: ChromaVectorDB):
    """Example search operations."""
    from openai_embeddings import embed_text
    
    # Example 1: Text search
    print("\n" + "="*50)
    print("Example 1: Text search")
    print("="*50)
    query = "a concert with bright lights"
    query_embedding = embed_text(query)
    results = db.search_by_text(query_embedding, n_results=5)
    
    print(f"\nQuery: '{query}'")
    print("\nTop 5 results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['filename']} (score: {result['score']:.4f})")
    
    # Example 2: Hybrid search (if you have image embeddings)
    # query_img_emb = embed_image("path/to/query/image.jpg")
    # results = db.hybrid_search(
    #     text_embedding=query_embedding,
    #     image_embedding=query_img_emb,
    #     alpha=0.7,
    #     n_results=5
    # )

if __name__ == "__main__":
    # Paths
    json_file = "precomputed_embeddings/openai_embeddings.json"
    
    # Migrate to Chroma
    db = migrate_json_to_chroma(json_file)
    
    # Example searches
    example_search(db)
    
    print("\n" + "="*50)
    print("Database ready! Use db.search_by_text() for queries.")
    print("="*50)

