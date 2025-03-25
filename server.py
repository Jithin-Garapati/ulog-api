"""
Server for the ULog Reasoning Agent API.
This file provides a clean entry point for running the API server.
"""
from flask import Flask
import os

# Import the Flask app from api.py
from api import app

if __name__ == '__main__':
    # Print a nice startup message
    print("=" * 50)
    print("ULog Analysis Agent API Server")
    print("=" * 50)
    print(f"API endpoint: http://localhost:5000/api")
    print(f"API documentation: http://localhost:5000/api/docs")
    print("=" * 50)
    
    # Start the server with auto-reloading disabled to prevent stream interruptions
    # The use_reloader=False is critical to prevent interrupting streaming responses
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
