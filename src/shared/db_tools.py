"""
Copyright (c) 2025 AI Hacks and Stacks
All rights reserved.

This file is part of the LLM Comparison Tool.

Database tools for managing and testing the vector database.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import time
from typing import List, Dict, Any

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.shared.db_providers import PostgresVectorProvider, get_db_provider
from src.shared.config import DB_CONFIG
from src.shared.logger import get_logger

logger = get_logger(__name__)

def get_random_embedding(dimension: int = 768) -> List[float]:
    """
    Generate a random embedding vector with the specified dimension.
    
    Args:
        dimension: The dimension of the embedding vector.
        
    Returns:
        A list of floats representing the embedding vector.
    """
    return np.random.rand(dimension).tolist()

def init_db(provider_name: str = "postgres") -> bool:
    """
    Initialize the database connection and create necessary tables.
    
    Args:
        provider_name: Name of the database provider to use.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Initializing database provider: {provider_name}")
    
    try:
        # Set environment variable for provider selection
        os.environ["DB_PROVIDER"] = provider_name
        
        # Get database provider
        db = get_db_provider()
        
        # Connect to the database
        success = db.connect()
        if not success:
            logger.error("Failed to connect to the database")
            return False
        
        logger.info("Successfully connected to the database")
        db.disconnect()
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        return False

def test_embeddings(args):
    """
    Test storing and retrieving embeddings.
    
    Args:
        args: Command line arguments.
    """
    db = get_db_provider()
    
    if not db.connect():
        logger.error("Failed to connect to database")
        return
    
    try:
        # Generate random embeddings for testing
        for i in range(args.num_vectors):
            # Generate a random vector with the specified dimension
            embedding = get_random_embedding(args.dimension)
            
            # Create unique doc ID and text
            doc_id = f"test-doc-{i}-{int(time.time())}"
            text = f"This is test document {i} with {args.dimension} dimensions."
            
            # Store embedding with model name
            success = db.store_embedding(
                doc_id=doc_id,
                text=text,
                embedding=embedding,
                metadata={"test": True, "dimension": args.dimension},
                embedding_model=f"test-model-{args.dimension}d"
            )
            
            if not success:
                logger.error(f"Failed to store embedding {i}")
                continue
                
            logger.info(f"Successfully stored embedding {i} with ID: {doc_id}")
            
            # Test similarity search with the same embedding (should return the document we just inserted)
            results = db.search_similar(embedding=embedding, top_k=1)
            
            if not results:
                logger.error(f"Failed to find similar documents for embedding {i}")
                continue
                
            # This should be the same document we just inserted
            result = results[0]
            logger.info(f"Found similar document: {result['id']} with similarity: {result['similarity']:.4f}")
            
            if 'embedding_model' in result:
                logger.info(f"Embedding model: {result['embedding_model']}, dimensions: {result.get('embedding_dimensions')}")
                
        # Check the total count
        count = db.count_embeddings()
        logger.info(f"Total embeddings in database: {count}")
            
    finally:
        db.disconnect()
        
def benchmark_database(args):
    """
    Benchmark database performance.
    
    Args:
        args: Command line arguments.
    """
    db = get_db_provider()
    
    if not db.connect():
        logger.error("Failed to connect to database")
        return
    
    try:
        # Generate benchmark data
        benchmark_data = []
        for i in range(args.num_vectors):
            doc_id = f"bench-doc-{i}-{int(time.time())}"
            embedding = get_random_embedding(args.dimension)
            benchmark_data.append({
                "id": doc_id,
                "text": f"Benchmark document {i}",
                "embedding": embedding,
                "metadata": {"benchmark": True, "index": i},
                "embedding_model": f"benchmark-model-{args.dimension}d"
            })
        
        # Benchmark batch insertion
        start_time = time.time()
        success = db.batch_store_embeddings(benchmark_data)
        batch_time = time.time() - start_time
        
        if success:
            logger.info(f"Batch stored {args.num_vectors} embeddings in {batch_time:.4f} seconds")
            logger.info(f"Average time per embedding (batch): {(batch_time / args.num_vectors) * 1000:.2f} ms")
        else:
            logger.error("Failed to batch store embeddings")
            
        # Benchmark individual insertion
        start_time = time.time()
        success_count = 0
        
        for i in range(min(args.num_vectors, 10)):  # Use a smaller set for individual inserts
            doc_id = f"bench-single-{i}-{int(time.time())}"
            embedding = get_random_embedding(args.dimension)
            
            if db.store_embedding(
                doc_id=doc_id,
                text=f"Single benchmark document {i}",
                embedding=embedding,
                metadata={"benchmark": True, "single": True, "index": i},
                embedding_model=f"benchmark-single-{args.dimension}d"
            ):
                success_count += 1
                
        single_time = time.time() - start_time
        
        if success_count > 0:
            logger.info(f"Individually stored {success_count} embeddings in {single_time:.4f} seconds")
            logger.info(f"Average time per embedding (individual): {(single_time / success_count) * 1000:.2f} ms")
        
        # Benchmark similarity search
        query_embedding = get_random_embedding(args.dimension)
        start_time = time.time()
        
        for i in range(args.num_searches):
            results = db.search_similar(embedding=query_embedding, top_k=10)
            
        search_time = time.time() - start_time
        logger.info(f"Performed {args.num_searches} similarity searches in {search_time:.4f} seconds")
        logger.info(f"Average time per search: {(search_time / args.num_searches) * 1000:.2f} ms")
        
    finally:
        db.disconnect()

def initialize_database(args):
    """
    Initialize the database.
    
    Args:
        args: Command line arguments.
    """
    logger.info("Initializing database...")
    
    # Connect to database
    db = get_db_provider()
    success = db.connect()
    
    if not success:
        logger.error("Failed to connect to database")
        return False
        
    try:
        # We'll store a test embedding to verify everything is working
        test_embedding = get_random_embedding(768)  # Use 768 as default dimension
        test_doc_id = f"init-test-{int(time.time())}"
        
        success = db.store_embedding(
            doc_id=test_doc_id,
            text="Database initialization test document",
            embedding=test_embedding,
            metadata={"init_test": True, "timestamp": time.time()},
            embedding_model="init-test-model"
        )
        
        if success:
            logger.info("Successfully stored test embedding")
            
            # Query it back to make sure search works
            results = db.search_similar(embedding=test_embedding, top_k=1)
            
            if results and len(results) > 0:
                logger.info("Successfully retrieved test embedding")
                logger.info(f"Similarity: {results[0]['similarity']:.4f}")
                
                if 'embedding_model' in results[0]:
                    logger.info(f"Model: {results[0]['embedding_model']}, dimensions: {results[0].get('embedding_dimensions')}")
                
                # Get total count
                count = db.count_embeddings()
                logger.info(f"Total embeddings in database: {count}")
                
                logger.info("Database initialized successfully!")
                return True
            else:
                logger.error("Failed to retrieve test embedding")
                return False
        else:
            logger.error("Failed to store test embedding")
            return False
            
    finally:
        db.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Database tools for the LLM Comparison Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize the database")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test database operations")
    test_parser.add_argument("--num-vectors", type=int, default=5, help="Number of test vectors (default: 5)")
    test_parser.add_argument("--dimension", type=int, default=768, help="Dimension of test vectors (default: 768)")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark database operations")
    benchmark_parser.add_argument("--num-vectors", type=int, default=100, help="Number of test vectors (default: 100)")
    benchmark_parser.add_argument("--dimension", type=int, default=768, help="Dimension of test vectors (default: 768)")
    benchmark_parser.add_argument("--num-searches", type=int, default=10, help="Number of searches to perform (default: 10)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "init":
        initialize_database(args)
    
    elif args.command == "test":
        test_embeddings(args)
    
    elif args.command == "benchmark":
        benchmark_database(args)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 