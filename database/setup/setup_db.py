#!/usr/bin/env python3
"""
Script to set up PostgreSQL database with pgvector extension and create tables.
This script will:
1. Connect to PostgreSQL
2. Enable pgvector extension
3. Create all necessary tables (users, images, faces)
4. Create indexes for performance
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

def get_db_config():
    """Get database configuration from environment variables."""
    config = {
        'dbname': os.getenv('DB_NAME', 'image_retrieval'),
        'user': os.getenv('DB_USER', 'imageuser'),
        'password': os.getenv('DB_PASSWORD', 'imagepass123'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
    }
    return config

def connect_to_postgres():
    """Connect to PostgreSQL server (without specifying database)."""
    config = get_db_config()
    try:
        # Connect to postgres database to create our database if needed
        conn = psycopg2.connect(
            dbname='postgres',
            user=config['user'],
            password=config['password'],
            host=config['host'],
            port=config['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        sys.exit(1)

def connect_to_db():
    """Connect to the application database."""
    config = get_db_config()
    try:
        conn = psycopg2.connect(
            dbname=config['dbname'],
            user=config['user'],
            password=config['password'],
            host=config['host'],
            port=config['port']
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    config = get_db_config()
    dbname = config['dbname']
    
    conn = connect_to_postgres()
    cur = conn.cursor()
    
    # Check if database exists
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (dbname,)
    )
    exists = cur.fetchone()
    
    if not exists:
        print(f"Creating database '{dbname}'...")
        cur.execute(f'CREATE DATABASE {dbname}')
        print(f"✓ Database '{dbname}' created")
    else:
        print(f"Database '{dbname}' already exists")
    
    cur.close()
    conn.close()

def setup_database():
    """Run the setup SQL script."""
    config = get_db_config()
    
    # Create database if needed
    create_database_if_not_exists()
    
    # Connect to the application database
    print(f"\nConnecting to database '{config['dbname']}'...")
    conn = connect_to_db()
    cur = conn.cursor()
    
    # Read and execute SQL script
    sql_file = Path(__file__).parent / 'init.sql'
    if not sql_file.exists():
        print(f"Error: SQL file not found: {sql_file}")
        sys.exit(1)
    
    print(f"Reading SQL script: {sql_file}")
    with open(sql_file, 'r') as f:
        sql_script = f.read()
    
    # Execute the script
    print("\nExecuting SQL script...")
    try:
        cur.execute(sql_script)
        conn.commit()
        print("✓ Database setup completed successfully!")
        
        # Verify tables were created
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        print(f"\nCreated tables: {', '.join([t[0] for t in tables])}")
        
        # Check if pgvector extension is enabled
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone():
            print("✓ pgvector extension is enabled")
        else:
            print("⚠ Warning: pgvector extension not found")
            
    except psycopg2.Error as e:
        print(f"Error executing SQL script: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    print("=" * 60)
    print("PostgreSQL Database Setup with pgvector")
    print("=" * 60)
    setup_database()
    print("\n" + "=" * 60)
    print("Setup complete! You can now use the database.")
    print("=" * 60)

