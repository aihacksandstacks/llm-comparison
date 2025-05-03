#!/usr/bin/env python
"""
Debug script to test the logging functionality.
Set the log level to DEBUG in .env to see all log messages.
"""

import os
import sys
import time
from src.shared.logger import get_logger, set_log_level
from src.shared.db_providers import get_db_provider

def main():
    """Test the logging functionality."""
    # Get a logger for this script
    logger = get_logger("debug_script")
    
    # Print welcome message
    logger.info("Starting logging debug script")
    
    # Force debug level for this script
    set_log_level("DEBUG")
    logger.info("Set log level to DEBUG for this script")
    
    # Log messages at different levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # Test database provider logging
    logger.info("Testing database provider logging")
    
    # Get database provider (will log which provider is being used)
    db_provider = get_db_provider()
    
    # Try to connect (will log connection details)
    logger.info("Attempting to connect to database")
    connection_successful = db_provider.connect()
    
    if connection_successful:
        logger.info("Connection to database successful")
        
        # Test storing an embedding (will log details about the embedding)
        logger.info("Testing store_embedding function")
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Simple test embedding
        db_provider.store_embedding(
            doc_id="test_doc_001",
            text="This is a test document for logging",
            embedding=test_embedding,
            metadata={"source": "debug_script", "timestamp": time.time()}
        )
        
        # Test searching embeddings (will log search details)
        logger.info("Testing search_similar function")
        results = db_provider.search_similar(test_embedding, top_k=3)
        logger.info(f"Found {len(results)} results")
        
        # Disconnect from database
        db_provider.disconnect()
    else:
        logger.error("Failed to connect to database")
    
    logger.info("Logging debug script completed")
    logger.info("Check the logs directory for the complete log file")

if __name__ == "__main__":
    main() 