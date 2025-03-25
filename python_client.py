#!/usr/bin/env python
"""
Python client for the ULog Reasoning Agent API.
This demonstrates how to use the API programmatically for parallel users.
"""

import requests
import json
import os
import time
from typing import Dict, Any, Optional, Generator, Union

class ULogAgentClient:
    """Client for interacting with the ULog Reasoning Agent API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize the ULog Agent Client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session_id = None
        self.ulog_file = None
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            session_id: The unique session ID
        """
        response = requests.post(f"{self.base_url}/api/session")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                self.session_id = data.get("session_id")
                return self.session_id
        
        response.raise_for_status()
        return None
    
    def upload_ulog(self, file_path: str, model: str = "qwen-qwq-32b") -> Dict[str, Any]:
        """
        Upload a ULog file for analysis.
        
        Args:
            file_path: Path to the ULog file
            model: Model to use for analysis
            
        Returns:
            Dict with upload status
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ULog file not found: {file_path}")
        
        self.ulog_file = file_path
        
        with open(file_path, "rb") as file:
            files = {"file": file}
            data = {"session_id": self.session_id, "model": model}
            response = requests.post(f"{self.base_url}/api/upload", files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        
        response.raise_for_status()
        return None
    
    def query(self, query_text: str, stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Send a query to the agent.
        
        Args:
            query_text: The query to send
            stream: If True, stream the response
            
        Returns:
            If stream=False: Dict with query response
            If stream=True: Generator yielding response chunks
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        data = {
            "session_id": self.session_id,
            "query": query_text,
            "stream": stream
        }
        
        if stream:
            response = requests.post(f"{self.base_url}/api/query", json=data, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line.decode('utf-8'))
            else:
                response.raise_for_status()
        else:
            response = requests.post(f"{self.base_url}/api/query", json=data)
            if response.status_code == 200:
                return response.json()
            response.raise_for_status()
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current session and clean up resources.
        
        Returns:
            Dict with end session status
        """
        if not self.session_id:
            raise ValueError("No active session. Call create_session() first.")
        
        data = {"session_id": self.session_id}
        response = requests.delete(f"{self.base_url}/api/session", json=data)
        
        if response.status_code == 200:
            result = response.json()
            self.session_id = None
            self.ulog_file = None
            return result
        
        response.raise_for_status()
        return None
    
    def __enter__(self):
        """Context manager entry."""
        if not self.session_id:
            self.create_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.session_id:
            self.end_session()


# Example usage
def main():
    """Example of using the ULog Agent Client."""
    # Replace with the URL of your deployed API
    api_url = "http://localhost:5000"  # or "https://your-railway-app.railway.app"
    
    # Path to your ULog file
    ulog_file_path = "path/to/your/ulog_file.ulg"
    
    # Create a client and start a session
    client = ULogAgentClient(base_url=api_url)
    
    try:
        # Using context manager to automatically handle session lifecycle
        with client:
            print(f"Session created: {client.session_id}")
            
            # Upload ULog file
            upload_result = client.upload_ulog(ulog_file_path)
            print(f"Upload result: {upload_result}")
            
            # Send a query with regular response
            query_result = client.query("What's the maximum altitude in this flight?", stream=False)
            print(f"Query result: {query_result['response']}")
            
            # Send another query with streaming response
            print("Streaming response:")
            for chunk in client.query("Analyze the throttle usage during takeoff", stream=True):
                if chunk.get("type") == "error":
                    print(f"Error: {chunk.get('content')}")
                else:
                    print(chunk.get("content", ""), end="", flush=True)
            
            print("\nSession completed successfully")
            
    except Exception as e:
        print(f"Error: {e}")
        # Make sure to clean up if there was an error
        if client.session_id:
            client.end_session()


# Parallel sessions example
def parallel_example():
    """Example of using multiple clients in parallel."""
    api_url = "http://localhost:5000"
    
    # Create multiple clients for different ULog files
    client1 = ULogAgentClient(base_url=api_url)
    client2 = ULogAgentClient(base_url=api_url)
    
    try:
        # Start sessions
        client1.create_session()
        client2.create_session()
        
        print(f"Session 1: {client1.session_id}")
        print(f"Session 2: {client2.session_id}")
        
        # Upload different ULog files
        client1.upload_ulog("path/to/first_flight.ulg")
        client2.upload_ulog("path/to/second_flight.ulg")
        
        # Query each session independently
        result1 = client1.query("What was the maximum speed?", stream=False)
        result2 = client2.query("What was the flight duration?", stream=False)
        
        print(f"Result from session 1: {result1['response']}")
        print(f"Result from session 2: {result2['response']}")
        
    finally:
        # Clean up both sessions
        if client1.session_id:
            client1.end_session()
        if client2.session_id:
            client2.end_session()


if __name__ == "__main__":
    main()
    # Uncomment to run the parallel example
    # parallel_example()
