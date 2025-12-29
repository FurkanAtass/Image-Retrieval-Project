#!/bin/bash
# Quick setup script for PostgreSQL database
# This script will DROP and recreate the database if it exists

echo "=========================================="
echo "PostgreSQL Database Setup Script"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Start PostgreSQL container
echo "Starting PostgreSQL container..."
docker compose up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Check if container is running
if ! docker ps | grep -q image_retrieval_db; then
    echo "Error: Container failed to start. Check logs with: docker compose logs"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

DB_NAME=${DB_NAME:-image_retrieval}
DB_USER=${DB_USER:-imageuser}
DB_PASSWORD=${DB_PASSWORD:-imagepass123}
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}

# Drop and recreate database
echo "Dropping existing database if it exists..."
PGPASSWORD=$DB_PASSWORD docker exec -i image_retrieval_db psql -U $DB_USER -d postgres <<EOF
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = '$DB_NAME'
  AND pid <> pg_backend_pid();

DROP DATABASE IF EXISTS $DB_NAME;
CREATE DATABASE $DB_NAME;
EOF

if [ $? -eq 0 ]; then
    echo "✓ Database '$DB_NAME' dropped and recreated"
else
    echo "⚠ Warning: Error dropping/recreating database (may not exist yet)"
fi

# Run Python setup script
echo ""
echo "Setting up database schema..."
python database/setup/setup_db.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Database is running at: localhost:5432"
echo "To connect: docker exec -it image_retrieval_db psql -U $DB_USER -d $DB_NAME"
echo "To stop: docker compose down"
echo ""

