"""
Helper functions for user management with password hashing.
Uses bcrypt for secure password hashing.
"""

import bcrypt
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'image_retrieval'),
        user=os.getenv('DB_USER', 'imageuser'),
        password=os.getenv('DB_PASSWORD', 'imagepass123'),
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432')
    )

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        Bcrypt hashed password string
    """
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        password: Plain text password to verify
        password_hash: Bcrypt hashed password from database
    
    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def create_user(email: str, password: str) -> Optional[int]:
    """
    Create a new user with hashed password.
    
    Args:
        email: User email (must be unique)
        password: Plain text password (will be hashed)
    
    Returns:
        User ID if successful, None if email already exists
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
            (email, password_hash)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        print(f"✓ User created with ID: {user_id}")
        return user_id
    except psycopg2.IntegrityError:
        conn.rollback()
        print(f"✗ Error: Email '{email}' already exists")
        return None
    except Exception as e:
        conn.rollback()
        print(f"✗ Error creating user: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Authenticate a user by email and password.
    
    Args:
        email: User email
        password: Plain text password
    
    Returns:
        User dict with id, email if successful, None if authentication fails
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s AND deleted_at IS NULL",
            (email,)
        )
        user = cur.fetchone()
        
        if user and verify_password(password, user['password_hash']):
            return {
                'id': user['id'],
                'email': user['email']
            }
        return None
    except Exception as e:
        print(f"✗ Error authenticating user: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get user by ID (without password hash)."""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute(
            "SELECT id, email, created_at FROM users WHERE id = %s AND deleted_at IS NULL",
            (user_id,)
        )
        user = cur.fetchone()
        return dict(user) if user else None
    except Exception as e:
        print(f"✗ Error getting user: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def soft_delete_user(user_id: int) -> bool:
    """
    Soft delete a user (sets deleted_at timestamp).
    
    Args:
        user_id: User ID to delete
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute(
            "UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = %s AND deleted_at IS NULL",
            (user_id,)
        )
        if cur.rowcount > 0:
            conn.commit()
            print(f"✓ User {user_id} soft deleted")
            return True
        else:
            print(f"✗ User {user_id} not found or already deleted")
            return False
    except Exception as e:
        conn.rollback()
        print(f"✗ Error deleting user: {e}")
        return False
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    # Example usage
    print("Creating test user...")
    user_id = create_user("test@example.com", "testpassword123")
    
    if user_id:
        print(f"\nAuthenticating user...")
        user = authenticate_user("test@example.com", "testpassword123")
        if user:
            print(f"✓ Authentication successful: {user}")
        else:
            print("✗ Authentication failed")
        
        print(f"\nGetting user by ID...")
        user_info = get_user_by_id(user_id)
        print(f"User info: {user_info}")

