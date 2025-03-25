"""
Main entry point for the multi-user ULog Reasoning Agent API.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes to register them with the app
from api_multiuser import app
import api_multiuser_routes

# Configure the R2 bucket name from environment
bucket_name = os.getenv('R2_BUCKET_NAME', 'qwen-agent-data')
os.environ['R2_BUCKET_NAME'] = bucket_name

if __name__ == '__main__':
    # Get configuration from environment variables
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    print(f"Starting ULog Reasoning Agent API (Multi-user version)")
    print(f"Debug mode: {debug}")
    print(f"Listening on {host}:{port}")
    
    # Start the Flask app
    app.run(debug=debug, host=host, port=port)
