-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create USERS table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Bcrypt hashed password
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL -- Soft delete
);

-- Create IMAGES table
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    description TEXT,
    tags JSONB DEFAULT '[]'::jsonb, -- Array of strings or objects with scores
    captured_at TIMESTAMP WITH TIME ZONE,
    location JSONB, -- {lat, lon} or {"name": "Vienna"} or null
    metadata JSONB DEFAULT '{}'::jsonb, -- Key-value map for camera, source, etc.
    image_embedding vector(512), -- CLIP image vector (adjust dimension as needed)
    text_embedding vector(512), -- Embedding of description (adjust dimension as needed)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL, -- Soft delete
    UNIQUE(user_id, file_path)
);

-- Create FACES table
CREATE TABLE IF NOT EXISTS faces (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    position JSONB NOT NULL, -- Normalized bbox: {x, y, w, h}
    face_embedding vector(512), -- Face embedding vector (adjust dimension as needed)
    person_id INTEGER, -- Nullable; filled if user labels or after clustering
    quality REAL, -- Optional: blur/pose score (0-1, higher is better)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL -- Soft delete
);

-- Create indexes for better query performance

-- Index on users for email lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE deleted_at IS NULL;

-- Index on images for user lookups
CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id) WHERE deleted_at IS NULL;

-- Index on images for file_path lookups
CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path) WHERE deleted_at IS NULL;

-- Index on images for captured_at (for time-based queries)
CREATE INDEX IF NOT EXISTS idx_images_captured_at ON images(captured_at) WHERE deleted_at IS NULL;

-- Vector similarity indexes for image embeddings (HNSW for fast approximate search)
-- Note: Partial indexes with WHERE deleted_at IS NULL for better performance
CREATE INDEX IF NOT EXISTS idx_images_image_embedding ON images 
    USING hnsw (image_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE deleted_at IS NULL;

-- Vector similarity indexes for text embeddings
CREATE INDEX IF NOT EXISTS idx_images_text_embedding ON images 
    USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE deleted_at IS NULL;

-- Index on faces for image lookups
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id) WHERE deleted_at IS NULL;

-- Index on faces for user lookups
CREATE INDEX IF NOT EXISTS idx_faces_user_id ON faces(user_id) WHERE deleted_at IS NULL;

-- Index on faces for person_id (for clustering/labeling)
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id) 
    WHERE person_id IS NOT NULL AND deleted_at IS NULL;

-- Vector similarity index for face embeddings
CREATE INDEX IF NOT EXISTS idx_faces_face_embedding ON faces 
    USING hnsw (face_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE deleted_at IS NULL;

-- GIN index for tags (JSONB array search)
CREATE INDEX IF NOT EXISTS idx_images_tags ON images USING GIN (tags) WHERE deleted_at IS NULL;

-- GIN index for metadata (JSONB key-value search)
CREATE INDEX IF NOT EXISTS idx_images_metadata ON images USING GIN (metadata) WHERE deleted_at IS NULL;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_images_updated_at BEFORE UPDATE ON images
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_faces_updated_at BEFORE UPDATE ON faces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE users IS 'User accounts for the image retrieval system';
COMMENT ON TABLE images IS 'Stored images with embeddings and metadata';
COMMENT ON TABLE faces IS 'Detected faces in images with embeddings';
COMMENT ON COLUMN users.email IS 'Unique email address for user authentication';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt hashed password (never store plain text passwords)';
COMMENT ON COLUMN users.deleted_at IS 'Soft delete timestamp (NULL = active, timestamp = deleted)';
COMMENT ON COLUMN images.tags IS 'Array of tag strings or objects with scores: ["tag1", "tag2"] or [{"tag": "tag1", "score": 0.9}]';
COMMENT ON COLUMN images.location IS 'Location data: {"lat": 48.2082, "lon": 16.3738} or {"name": "Vienna"}';
COMMENT ON COLUMN images.deleted_at IS 'Soft delete timestamp (NULL = active, timestamp = deleted)';
COMMENT ON COLUMN faces.position IS 'Normalized bounding box: {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}';
COMMENT ON COLUMN faces.quality IS 'Face quality score 0-1, higher is better (blur, pose, etc.)';
COMMENT ON COLUMN faces.deleted_at IS 'Soft delete timestamp (NULL = active, timestamp = deleted)';

