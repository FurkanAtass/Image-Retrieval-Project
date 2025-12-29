# PostgreSQL Database Setup

This directory contains scripts to set up PostgreSQL with pgvector extension for the image retrieval system.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+
- Required Python packages: `psycopg2-binary`, `python-dotenv`, `bcrypt`

## Quick Start

### 1. Install Dependencies

```bash
pip install psycopg2-binary python-dotenv bcrypt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and update the database credentials if needed:

```bash
cp .env.example .env
```

The default configuration in `.env.example` matches the Docker setup:
- Database: `image_retrieval`
- User: `imageuser`
- Password: `imagepass123`
- Host: `localhost`
- Port: `5432`

### 3. Start PostgreSQL with Docker

```bash
docker compose up -d
```

This will start PostgreSQL with pgvector extension on port 5432.

### 4. Set Up Database Schema

Run the setup script to create tables and indexes:

```bash
python database/setup/setup_db.py
```

Or use the convenient script:

```bash
./database/setup/init_db.sh
```

## Database Schema

### Tables

1. **users**
   - `id` (SERIAL PRIMARY KEY)
   - `email` (VARCHAR, UNIQUE) - User email for authentication
   - `password_hash` (VARCHAR) - Bcrypt hashed password
   - `created_at`, `updated_at` (timestamps)
   - `deleted_at` (TIMESTAMP, nullable) - Soft delete timestamp

2. **images**
   - `id` (SERIAL PRIMARY KEY)
   - `user_id` (INTEGER, FK to users)
   - `file_path` (VARCHAR) - path/filename/object key to the image file
   - `description` (TEXT) - generated caption
   - `tags` (JSONB) - array of tag strings or objects with scores
   - `captured_at` (TIMESTAMP) - date/time from EXIF or filesystem
   - `location` (JSONB) - {lat, lon} or {"name": "Vienna"}
   - `metadata` (JSONB) - key-value map for camera, source, etc.
   - `image_embedding` (vector(512)) - CLIP image vector
   - `text_embedding` (vector(512)) - embedding of description
   - `created_at`, `updated_at` (timestamps)
   - `deleted_at` (TIMESTAMP, nullable) - Soft delete timestamp

3. **faces**
   - `id` (SERIAL PRIMARY KEY)
   - `user_id` (INTEGER, FK to users)
   - `image_id` (INTEGER, FK to images)
   - `position` (JSONB) - normalized bbox: {x, y, w, h}
   - `face_embedding` (vector(512)) - face embedding vector
   - `person_id` (INTEGER, nullable) - for clustering/labeling
   - `quality` (REAL) - blur/pose score (0-1)
   - `created_at`, `updated_at` (timestamps)
   - `deleted_at` (TIMESTAMP, nullable) - Soft delete timestamp

### Indexes

- Vector similarity indexes (HNSW) on all embedding columns (with soft delete filtering)
- Indexes on foreign keys and commonly queried fields
- GIN indexes on JSONB columns (tags, metadata) for efficient JSON queries
- Partial indexes exclude soft-deleted records for better performance

### Soft Delete

All tables include `deleted_at` columns for soft delete functionality:
- `NULL` = active record
- `TIMESTAMP` = deleted record

Queries should filter with `WHERE deleted_at IS NULL` to exclude deleted records.

### User Authentication

Users table includes:
- `email` (unique) - for login
- `password_hash` - bcrypt hashed password (never store plain text)

See `setup/user_helpers.py` for password hashing and authentication functions.

## Setup Files

All setup-related files are located in the `database/setup/` directory:

- `init.sql` - Database schema SQL script with all table definitions
- `setup_db.py` - Python script to initialize the database
- `init_db.sh` - Bash script for quick setup (drops and recreates database)
- `user_helpers.py` - Helper functions for user management (password hashing, authentication)

## Usage

### Connection String

You can connect using either individual environment variables or a connection string:

```python
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Using individual variables
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)

# Or using connection string
DATABASE_URL = os.getenv('DATABASE_URL')
conn = psycopg2.connect(DATABASE_URL)
```

### Example Queries

#### Similarity Search

```sql
-- Find similar images using text embedding
SELECT id, file_path, description,
       1 - (text_embedding <=> %s) AS similarity
FROM images
WHERE user_id = 1
ORDER BY text_embedding <=> %s
LIMIT 10;
```

#### Hybrid Search (text + image)

```sql
-- Combine text and image embeddings
SELECT id, file_path, description,
       0.7 * (1 - (text_embedding <=> %s)) + 
       0.3 * (1 - (image_embedding <=> %s)) AS similarity
FROM images
WHERE user_id = 1
ORDER BY similarity DESC
LIMIT 10;
```

## Docker Commands

```bash
# Start database
docker compose up -d

# Stop database
docker compose down

# View logs
docker compose logs -f

# Connect to database with psql
docker exec -it image_retrieval_db psql -U imageuser -d image_retrieval

# Stop and remove all data
docker compose down -v
```

## Troubleshooting

### Connection Refused

Make sure Docker is running and the container is up:
```bash
docker compose ps
```

### pgvector Extension Not Found

The Docker image `pgvector/pgvector:pg16` includes pgvector. If you're using a different image, install it manually:

```sql
CREATE EXTENSION vector;
```

### Port Already in Use

If port 5432 is already in use, change it in `docker-compose.yml` and update `.env`:

```yaml
ports:
  - "5433:5432"  # Use 5433 instead
```

## Next Steps

See `../vector_db_postgres.py` for a Python wrapper class that simplifies working with this database schema.

