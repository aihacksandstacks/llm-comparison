#!/bin/bash
# Script to rebuild and restart the Docker containers

echo "Stopping and removing existing containers..."
docker-compose -f docker-compose.dev.yml down

echo "Building new image with Playwright browsers..."
docker-compose -f docker-compose.dev.yml build

echo "Starting containers..."
docker-compose -f docker-compose.dev.yml up 