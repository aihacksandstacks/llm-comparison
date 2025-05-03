#!/usr/bin/env python
"""
Development runner for LLM Comparison Tool.
Launches the Streamlit app with hot reloading enabled.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from src.shared.logger import get_logger, set_log_level

logger = get_logger("run_dev")

def setup_env():
    """Set up environment variables for development."""
    # Force debug log level
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "true"
    os.environ["STREAMLIT_CLIENT_TOOLBAR_MODE"] = "developer"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_RUNNER_FAST_RERUNS"] = "true"
    os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
    os.environ["PYTHONUNBUFFERED"] = "1"

def find_streamlit_app():
    """Find the main Streamlit app file."""
    app_path = Path("src/features/base_ui/app.py")
    if not app_path.exists():
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)
    return app_path

def main():
    """Run the development server."""
    logger.info("Starting LLM Comparison Tool in development mode")
    
    # Setup environment for development
    setup_env()
    
    # Find the app
    app_path = find_streamlit_app()
    logger.info(f"Found Streamlit app at {app_path}")
    
    # Install watchdog for file monitoring if not already installed
    try:
        import watchdog
        logger.info("Watchdog is already installed")
    except ImportError:
        logger.info("Installing watchdog for file monitoring")
        subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"])
    
    # Launch Streamlit with hot reloading
    logger.info("Launching Streamlit with hot reloading")
    cmd = [
        "streamlit", "run", str(app_path),
        "--server.runOnSave=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run Streamlit with hot reloading
        process = subprocess.Popen(cmd)
        
        # Wait for process to complete or user interrupt
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping development server")
                process.terminate()
                process.wait()
                break
    except Exception as e:
        logger.error(f"Error running development server: {e}")
        sys.exit(1)
    
    logger.info("Development server stopped")

if __name__ == "__main__":
    main() 