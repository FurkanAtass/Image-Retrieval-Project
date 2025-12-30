# Image Retrieval API

FastAPI application for searching and retrieving images using vector similarity search.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the database is set up and embeddings are loaded:
```bash
./database/setup/init_db.sh
```

3. Start the API server:


```bash
python api/main.py
# Or using uvicorn directly:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**Access Swagger UI (Interactive API Documentation):**
- Swagger UI: `http://localhost:8000/docs`
- API root: `http://localhost:8000/`

## API Endpoints

### GET `/`
Root endpoint with API information.

### GET `/health`
Health check endpoint.

### POST `/search`
General search endpoint supporting text, image, or hybrid queries.

**Request Body:**
```json
{
  "query": "person walking in the city",
  "query_image_path": null,
  "user_id": 1,
  "embedding_type": "text",
  "alpha": 0.7,
  "min_score": 0.3,
  "gap_threshold": 0.15,
  "max_results": 50
}
```

**Response:**
```json
{
  "results": [
    {
      "id": 1,
      "file_path": "city_walk_people.jpg",
      "description": "...",
      "score": 0.89
    }
  ],
  "total_found": 15,
  "returned": 5,
  "query_type": "text"
}
```

### GET `/search/text`
Search images by text query.

**Query Parameters:**
- `query` (required): Text query to search for
- `user_id` (default: 1): User ID
- `min_score` (default: 0.3): Minimum similarity score
- `gap_threshold` (default: 0.15): Gap threshold for smart selection
- `max_results` (default: 50): Maximum results to consider

**Example:**
```
GET /search/text?query=person%20walking&min_score=0.4
```

### GET `/search/image`
Search images by image query.

**Query Parameters:**
- `query_image_path` (required): Path to query image file
- `user_id` (default: 1): User ID
- `min_score` (default: 0.3): Minimum similarity score
- `gap_threshold` (default: 0.15): Gap threshold for smart selection
- `max_results` (default: 50): Maximum results to consider

**Example:**
```
GET /search/image?query_image_path=images/query.jpg&min_score=0.5
```

## Smart Scoring

The API uses intelligent scoring to determine how many results to return:

1. **Minimum Score Filter**: Results must have a similarity score >= `min_score` (default: 0.3)
2. **Gap Detection**: If there's a significant gap (>= `gap_threshold`) between consecutive scores, it stops before the gap
   - Example: Scores [0.9, 0.8, 0.2, 0.19] with gap_threshold=0.15
   - Returns first 2 results (before the gap between 0.8 and 0.2)

This ensures that only relevant results are returned, automatically filtering out low-quality matches.

