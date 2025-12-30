#!/usr/bin/env python3
"""
Load embeddings from JSON file into PostgreSQL database.
Creates admin user if it doesn't exist, then loads all embeddings.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

# Import user_helpers from same directory
from user_helpers import create_user, get_db_connection

load_dotenv()

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'dbname': os.getenv('DB_NAME', 'image_retrieval'),
        'user': os.getenv('DB_USER', 'imageuser'),
        'password': os.getenv('DB_PASSWORD', 'imagepass123'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
    }

def get_or_create_admin_user():
    """Get admin user ID, creating it if it doesn't exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Try to get existing admin user
        try:
            cur.execute("SELECT id FROM users WHERE email = %s AND deleted_at IS NULL", ('admin@example.com',))
            result = cur.fetchone()
            if result:
                user_id = result[0]
                print(f"✓ Found existing admin user (ID: {user_id})")
                return user_id
        except Exception as e:
            print(f"Error checking for admin user: {e}")
        
        # Create admin user if it doesn't exist
        print("Creating admin user...")
        user_id = create_user('admin@example.com', 'admin123')
        if user_id:
            return user_id
        else:
            # If create_user failed, try to get it anyway (might have been created concurrently)
            cur.execute("SELECT id FROM users WHERE email = %s AND deleted_at IS NULL", ('admin@example.com',))
            result = cur.fetchone()
            if result:
                return result[0]
    except Exception as e:
        print(f"Error getting or creating admin user: {e}")
        raise Exception("Failed to create or find admin user")
    finally:
        cur.close()
        conn.close()

def load_embeddings_from_json(json_file: str, user_id: int, images_dir: str = 'dataset/images'):
    """
    Load embeddings from JSON file into database.
    
    Args:
        json_file: Path to the embeddings JSON file
        user_id: User ID to associate images with
        images_dir: Directory where images are stored (default: 'dataset/images')
    """
    json_path = Path(json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {json_file}")
    
    # Get project root (parent of database directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    images_path = project_root / images_dir
    
    print(f"\nLoading embeddings from {json_file}...")
    print(f"Images directory: {images_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    
    print(f"Found {len(embeddings)} embeddings to load")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Prepare data for bulk insert
        images_data = []
        for emb in embeddings:
            if not emb.get('text_embedding') or not emb.get('filename'):
                print(f"⚠ Skipping incomplete embedding: {emb.get('filename', 'unknown')}")
                continue
            
            # Construct full file path: images_dir/filename
            # Store relative path from project root (e.g., "dataset/images/filename.jpg")
            filename = emb['filename']
            file_path = f"{images_dir}/{filename}"
            
            # Convert embeddings to list format for PostgreSQL vector type
            text_emb = emb['text_embedding']
            if isinstance(text_emb, list):
                text_emb_list = text_emb
            else:
                text_emb_list = list(text_emb) if hasattr(text_emb, '__iter__') else [text_emb]
            
            image_emb_list = None
            if emb.get('image_embedding'):
                image_emb = emb['image_embedding']
                if isinstance(image_emb, list):
                    image_emb_list = image_emb
                else:
                    image_emb_list = list(image_emb) if hasattr(image_emb, '__iter__') else [image_emb]
            
            images_data.append((
                user_id,
                file_path,
                emb.get('description', ''),
                json.dumps(emb.get('tags', [])),  # JSONB
                None,  # captured_at
                None,  # location
                json.dumps(emb.get('metadata', {})),  # JSONB metadata
                text_emb_list,  # text_embedding (list)
                image_emb_list  # image_embedding (list or None)
            ))
        
        print(f"\nInserting {len(images_data)} images into database...")
        
        # Insert one by one (pgvector accepts arrays directly)
        insert_query = """
            INSERT INTO images (
                user_id, file_path, description, tags, captured_at, 
                location, metadata, text_embedding, image_embedding
            ) VALUES (%s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s::vector, %s::vector)
            ON CONFLICT (user_id, file_path) DO UPDATE SET
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                metadata = EXCLUDED.metadata,
                text_embedding = EXCLUDED.text_embedding,
                image_embedding = EXCLUDED.image_embedding,
                updated_at = CURRENT_TIMESTAMP
        """
        
        inserted = 0
        for data in images_data:
            try:
                cur.execute(insert_query, data)
                inserted += 1
                if inserted % 10 == 0:
                    print(f"  Inserted {inserted}/{len(images_data)} images...", end='\r')
            except Exception as e:
                print(f"\n⚠ Error inserting {data[1]}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n  Inserted {inserted}/{len(images_data)} images")
        
        conn.commit()
        print(f"✓ Successfully loaded {len(images_data)} images into database")
        
        # Verify count
        cur.execute("SELECT COUNT(*) FROM images WHERE user_id = %s AND deleted_at IS NULL", (user_id,))
        count = cur.fetchone()[0]
        print(f"✓ Total images in database for user {user_id}: {count}")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error loading embeddings: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to load embeddings."""
    # Get project root (parent of database directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    json_file = project_root / 'precomputed_embeddings' / 'openai_embeddings.json'
    
    # Images directory relative to project root
    images_dir = 'dataset/images'  # Can be changed via command line argument or config
    
    print("=" * 60)
    print("Loading Embeddings into PostgreSQL")
    print("=" * 60)
    
    # Get or create admin user
    user_id = get_or_create_admin_user()
    
    # Load embeddings with images directory path
    load_embeddings_from_json(str(json_file), user_id, images_dir=images_dir)
    
    print("\n" + "=" * 60)
    print("✓ Embeddings loaded successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()

