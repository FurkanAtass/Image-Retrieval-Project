"""
FastAPI application for image search and retrieval.
Provides endpoints for searching images using text, image, or hybrid queries.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai_embeddings import embed_text, embed_image

# Load .env from project root
load_dotenv(dotenv_path=project_root / '.env')

app = FastAPI(title="Image Retrieval API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'image_retrieval'),
        user=os.getenv('DB_USER', 'imageuser'),
        password=os.getenv('DB_PASSWORD', 'imagepass123'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )

def smart_threshold_selection(scores: List[float], min_score: float = 0.2, gap_threshold: float = 0.15) -> int:
    """
    Intelligently select how many results to return based on scores.
    
    Strategy:
    1. Filter results with score >= min_score
    2. Detect significant gaps in scores (indicating drop-off in relevance)
    3. Return results before the first significant gap
    
    Args:
        scores: List of similarity scores (sorted descending)
        min_score: Minimum score threshold (default: 0.3)
        gap_threshold: Minimum gap size to consider significant (default: 0.15)
    
    Returns:
        Number of results to return
    """
    if not scores:
        return 0
    
    # Filter by minimum score
    valid_scores = [s for s in scores if s >= min_score]
    if not valid_scores:
        return 0
    
    # If only one valid result, return it
    if len(valid_scores) == 1:
        return 1
    
    # Detect significant gaps
    for i in range(len(valid_scores) - 1):
        gap = valid_scores[i] - valid_scores[i + 1]
        if gap >= gap_threshold:
            # Significant gap detected, return results up to this point
            return i + 1
    
    # No significant gap found, return all valid results
    return len(valid_scores)

# Request/Response models
class SearchRequest(BaseModel):
    query: Optional[str] = None
    query_image_path: Optional[str] = None
    user_id: int = 1  # Default to admin user
    alpha: float = 0.7  # For hybrid search (0.0 = image only, 1.0 = text only)
    min_score: Optional[float] = 0.3  # Minimum similarity score
    gap_threshold: Optional[float] = 0.15  # Gap threshold for smart selection
    max_results: Optional[int] = 50  # Maximum results to consider

class SearchResult(BaseModel):
    file_path: str
    score: float
    id: int

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_found: int
    returned: int
    query_type: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Image Retrieval API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "search_text": "/search/text",
            "search_image": "/search/image",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

@app.get("/search", response_model=SearchResponse)
@app.post("/search", response_model=SearchResponse)
async def search_images(
    request: SearchRequest = None,
    # GET parameters (optional, for GET requests)
    query: Optional[str] = Query(None, description="Text query to search for"),
    query_image_path: Optional[str] = Query(None, description="Path to query image file"),
    user_id: int = Query(1, description="User ID"),
    alpha: float = Query(0.7, description="For hybrid search (0.0 = image only, 1.0 = text only)"),
    min_score: Optional[float] = Query(0.3, description="Minimum similarity score"),
    gap_threshold: Optional[float] = Query(0.15, description="Gap threshold for smart selection"),
    max_results: Optional[int] = Query(50, description="Maximum results to consider")
):
    """
    General search endpoint supporting text, image, or hybrid queries.
    Automatically determines the embedding type based on provided inputs:
    - Only text query → "text"
    - Only image query → "image"
    - Both queries → "fuse"
    """
    if not request.query and not request.query_image_path:
        raise HTTPException(status_code=400, detail="Either 'query' (text) or 'query_image_path' must be provided")
    
    # Automatically determine embedding type based on provided inputs
    has_text = bool(request.query and request.query.strip())
    has_image = bool(request.query_image_path and request.query_image_path.strip())
    
    if has_text and has_image:
        embedding_type = "fuse"
    elif has_image:
        embedding_type = "image"
    else:  # has_text
        embedding_type = "text"
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get query embeddings
        text_embedding = None
        image_embedding = None
        
        if has_text:
            text_emb = embed_text(request.query)
            # Convert numpy array or tensor to list
            if hasattr(text_emb, 'tolist'):
                text_embedding = text_emb.tolist()
            elif isinstance(text_emb, (list, tuple)):
                text_embedding = list(text_emb)
            else:
                text_embedding = text_emb
        
        if has_image:
            image_path = Path(request.query_image_path)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {request.query_image_path}")
            image_emb = embed_image(str(image_path))
            # Convert tensor or numpy array to list
            if hasattr(image_emb, 'tolist'):
                image_embedding = image_emb.tolist()
            elif isinstance(image_emb, (list, tuple)):
                image_embedding = list(image_emb)
            else:
                image_embedding = image_emb
        
        # Build query based on determined embedding type
        if embedding_type == "text":
            query_sql = """
                SELECT id, file_path,
                       1 - (text_embedding <=> %s::vector) AS score
                FROM images
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY text_embedding <=> %s::vector
                LIMIT %s
            """
            params = (text_embedding, request.user_id, text_embedding, request.max_results)
        
        elif embedding_type == "image":
            query_sql = """
                SELECT id, file_path,
                       1 - (image_embedding <=> %s::vector) AS score
                FROM images
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY image_embedding <=> %s::vector
                LIMIT %s
            """
            params = (image_embedding, request.user_id, image_embedding, request.max_results)
        
        else:  # fuse/hybrid
            # For hybrid, use a weighted combination
            query_sql = """
                SELECT id, file_path,
                       %s * (1 - (text_embedding <=> %s::vector)) + 
                       %s * (1 - (image_embedding <=> %s::vector)) AS score
                FROM images
                WHERE user_id = %s AND deleted_at IS NULL
                ORDER BY score DESC
                LIMIT %s
            """
            alpha = request.alpha
            params = (alpha, text_embedding, 1-alpha, image_embedding, request.user_id, request.max_results)
        
        # Execute query
        cur.execute(query_sql, params)
        results = cur.fetchall()
        
        # Extract scores and apply smart threshold
        scores = [float(r['score']) for r in results]
        num_to_return = smart_threshold_selection(
            scores,
            min_score=request.min_score or 0.3,
            gap_threshold=request.gap_threshold or 0.15
        )
        
        # Return selected results
        selected_results = results[:num_to_return]
        
        return SearchResponse(
            results=[
                SearchResult(
                    id=r['id'],
                    file_path=r['file_path'],
                    score=float(r['score'])
                )
                for r in selected_results
            ],
            total_found=len(results),
            returned=num_to_return,
            query_type=embedding_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.get("/search/text", response_model=SearchResponse)
async def search_by_text(
    query: str = Query(..., description="Text query to search for"),
    user_id: int = Query(1, description="User ID"),
    min_score: Optional[float] = Query(0.3, description="Minimum similarity score"),
    gap_threshold: Optional[float] = Query(0.15, description="Gap threshold for smart selection"),
    max_results: Optional[int] = Query(50, description="Maximum results to consider")
):
    """Search images by text query."""
    request = SearchRequest(
        query=query,
        user_id=user_id,
        min_score=min_score,
        gap_threshold=gap_threshold,
        max_results=max_results
    )
    return await search_images(request)

@app.get("/search/image", response_model=SearchResponse)
async def search_by_image(
    query_image_path: str = Query(..., description="Path to query image"),
    user_id: int = Query(1, description="User ID"),
    min_score: Optional[float] = Query(0.3, description="Minimum similarity score"),
    gap_threshold: Optional[float] = Query(0.15, description="Gap threshold for smart selection"),
    max_results: Optional[int] = Query(50, description="Maximum results to consider")
):
    """Search images by image query."""
    request = SearchRequest(
        query_image_path=query_image_path,
        user_id=user_id,
        min_score=min_score,
        gap_threshold=gap_threshold,
        max_results=max_results
    )
    return await search_images(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

