#!/bin/bash
# Development script for LLM Comparison Tool
# Starts the application in development mode with hot reloading

echo "Starting LLM Comparison Tool in development mode..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose -f docker-compose.dev.yml down

# Start the development environment
echo "Starting development environment with hot reloading..."
docker-compose -f docker-compose.dev.yml up -d

# Show logs for webapp container
echo "Showing logs for webapp container (Ctrl+C to exit logs, app will keep running)..."
docker-compose -f docker-compose.dev.yml logs -f webapp 