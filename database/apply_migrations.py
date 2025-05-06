#!/usr/bin/env python
"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Script to apply database migrations.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import psycopg2
from dotenv import load_dotenv
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_migrations')

def get_db_connection():
    """Get database connection using environment variables."""
    db_params = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'dbname': os.getenv('POSTGRES_DB', 'llm_comparison')
    }
    
    try:
        logger.info(f"Connecting to PostgreSQL at {db_params['host']}:{db_params['port']}/{db_params['dbname']}")
        conn = psycopg2.connect(**db_params)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def create_migrations_table(conn):
    """Create the migrations table if it doesn't exist."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    migration_name TEXT NOT NULL UNIQUE,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        logger.info("Migrations table created or already exists")
        return True
    except Exception as e:
        logger.error(f"Error creating migrations table: {e}")
        conn.rollback()
        return False

def get_applied_migrations(conn):
    """Get list of already applied migrations."""
    applied = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT migration_name FROM migrations ORDER BY id")
            applied = [row[0] for row in cur.fetchall()]
        logger.info(f"Found {len(applied)} already applied migrations")
        return applied
    except Exception as e:
        logger.error(f"Error getting applied migrations: {e}")
        return []

def apply_migration(conn, migration_path, migration_name, dry_run=False):
    """Apply a single migration file."""
    try:
        # Read migration file
        with open(migration_path, 'r') as f:
            sql = f.read()
        
        logger.info(f"Applying migration: {migration_name}")
        
        if dry_run:
            logger.info("[DRY RUN] Would execute SQL:")
            logger.info(sql[:500] + "..." if len(sql) > 500 else sql)
            return True
        
        # Execute migration
        with conn.cursor() as cur:
            cur.execute(sql)
            
            # Record migration
            cur.execute(
                "INSERT INTO migrations (migration_name) VALUES (%s)",
                (migration_name,)
            )
        
        conn.commit()
        logger.info(f"Successfully applied migration: {migration_name}")
        return True
    except Exception as e:
        logger.error(f"Error applying migration {migration_name}: {e}")
        conn.rollback()
        return False

def apply_migrations(migrations_dir, dry_run=False):
    """Apply all pending migrations."""
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logger.error("Could not connect to database. Exiting.")
        return False
    
    try:
        # Create migrations table if it doesn't exist
        if not create_migrations_table(conn):
            logger.error("Could not create migrations table. Exiting.")
            return False
        
        # Get already applied migrations
        applied_migrations = get_applied_migrations(conn)
        
        # Get all migration files
        migration_files = sorted([f for f in Path(migrations_dir).glob('*.sql')])
        logger.info(f"Found {len(migration_files)} migration files")
        
        # Filter migrations that need to be applied
        pending_migrations = []
        for migration_path in migration_files:
            migration_name = migration_path.name
            if migration_name not in applied_migrations:
                pending_migrations.append((migration_path, migration_name))
        
        # Sort migrations by name to ensure proper order
        # Assumes migrations are named with numeric prefixes like 001_name.sql
        pending_migrations.sort(key=lambda x: x[1])
        
        logger.info(f"Found {len(pending_migrations)} pending migrations to apply")
        
        # Apply each pending migration
        success = True
        for migration_path, migration_name in pending_migrations:
            if not apply_migration(conn, migration_path, migration_name, dry_run):
                success = False
                logger.error(f"Failed to apply migration: {migration_name}")
                break
        
        return success
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Apply database migrations")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print migrations that would be applied without executing them"
    )
    parser.add_argument(
        "--migrations-dir", 
        default=str(Path(__file__).parent / "migrations"),
        help="Directory containing migration files"
    )
    
    args = parser.parse_args()
    
    # Ensure migrations directory exists
    migrations_dir = Path(args.migrations_dir)
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return 1
    
    # Apply migrations
    if apply_migrations(migrations_dir, args.dry_run):
        logger.info("All migrations applied successfully!")
        return 0
    else:
        logger.error("Failed to apply all migrations")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 