"""
Multi-user API server for the ULog Reasoning Agent.
Provides a web API with proper session management for multiple parallel users.
"""
from flask import Flask, request, jsonify, Response, stream_with_context, render_template, send_from_directory
from flask_cors import CORS
import json
import uuid
import time
import threading
import os
import glob
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional

from agent import ULogReasoningAgent
from config import CONFIG
from storage import storage
from session_manager import session_manager
from auth import require_auth, optional_auth

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'ulg'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class StreamCollector:
    """Collects streaming responses from the ULog agent and formats them for the API."""
    
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self.messages = []  # Store all messages
        self.current_message = {"type": "none", "content": ""}
        self.final_answer = None
        
    def flush_current_message(self):
        """Flush current message to messages list."""
        if self.current_message["type"] != "none":
            # Add a unique ID and timestamp to the message
            message = self.current_message.copy()
            message["id"] = str(uuid.uuid4())
            message["timestamp"] = time.time()
            self.messages.append(message)
            
            # Add to session in session manager
            session_manager.add_message_to_session(self.session_id, message)
            
            self.current_message = {"type": "none", "content": ""}
    
    def on_thinking(self, content: str):
        """Handle thinking/reasoning content from the agent."""
        if self.current_message["type"] != "thinking":
            self.flush_current_message()
            self.current_message = {"type": "thinking", "content": ""}
        self.current_message["content"] += content
    
    def on_tool_call(self, tool_call: Dict):
        """Handle a tool call from the agent."""
        self.flush_current_message()
        
        # Extract tool name and arguments, handling different possible formats
        tool_name = "unknown_tool"
        tool_args = {}
        
        # Handle different tool call formats
        if isinstance(tool_call, dict):
            # Direct dictionary format
            if "name" in tool_call:
                tool_name = tool_call["name"]
            elif "tool_name" in tool_call:
                tool_name = tool_call["tool_name"]
            elif "function" in tool_call and "name" in tool_call["function"]:
                tool_name = tool_call["function"]["name"]
                
            # Extract arguments
            if "arguments" in tool_call:
                tool_args = tool_call["arguments"]
            elif "args" in tool_call:
                tool_args = tool_call["args"]
            elif "function" in tool_call and "arguments" in tool_call["function"]:
                tool_args = tool_call["function"]["arguments"]
        
        # Log the actual tool call structure for debugging
        print(f"DEBUG - Tool call structure: {tool_call}")
        
        message = {
            "type": "tool_call",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "tool_name": tool_name,
            "args": tool_args
        }
        
        self.messages.append(message)
        
        # Add to session in session manager
        session_manager.add_message_to_session(self.session_id, message)
    
    def on_tool_result(self, tool_result: Dict):
        """Handle result of a tool call."""
        # Note: We use the tool_result data structure from the agent directly
        message = {
            "type": "tool_result",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "result": tool_result.get("content", "")
        }
        
        self.messages.append(message)
        
        # Add to session in session manager
        session_manager.add_message_to_session(self.session_id, message)
    
    def on_final_answer(self, final_answer: str):
        """Handle the final answer from the agent."""
        self.flush_current_message()
        self.final_answer = final_answer
        
        message = {
            "type": "final_answer",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "content": final_answer
        }
        
        self.messages.append(message)
        
        # Add to session in session manager
        session_manager.add_message_to_session(self.session_id, message)
        
        # Set final answer in session manager
        session_manager.set_final_answer(self.session_id, final_answer)
        
    def updates(self):
        """Generate updates from the collector."""
        # Get session from session manager
        session = session_manager.get_chat_session(self.session_id, self.user_id)
        if not session:
            yield {"type": "error", "error": "Session not found"}
            return
        
        # Yield existing messages from the session
        for message in session["messages"]:
            yield message
        
        # Track the last message index
        last_message_index = len(session["messages"])
        
        # Keep yielding new messages as they come in
        while True:
            # Get updated session
            session = session_manager.get_chat_session(self.session_id, self.user_id)
            if not session:
                yield {"type": "error", "error": "Session not found"}
                break
            
            # Check if there are new messages
            if len(session["messages"]) > last_message_index:
                # Yield new messages
                for i in range(last_message_index, len(session["messages"])):
                    yield session["messages"][i]
                last_message_index = len(session["messages"])
            
            # Check if session is done
            if session["status"] == "completed" and session["final_answer"] is not None:
                # Send completion message
                yield {
                    "type": "complete",
                    "final_answer": session["final_answer"]
                }
                break
            
            # Check if session has an error
            if session["status"] == "error" and session["error"] is not None:
                # Send error message
                yield {
                    "type": "error",
                    "error": session["error"]
                }
                break
            
            # Small sleep to avoid tight loop
            time.sleep(0.1)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the session."""
        # Get session from session manager
        session = session_manager.get_chat_session(self.session_id, self.user_id)
        if not session:
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "error": "Session not found"
            }
        
        return session

class ULogAgentSession:
    """Manages a session with the ULog agent."""
    
    def __init__(self, session_id: str, user_id: str, query: str, show_reasoning: bool = True, 
                 model: str = None, ulog_file: str = None):
        """Initialize a new session."""
        self.session_id = session_id
        self.user_id = user_id
        self.query = query
        self.show_reasoning = show_reasoning
        self.model = model or CONFIG["agent"]["model"]
        self.ulog_file = ulog_file  # Store the ULog file path
        self.agent = ULogReasoningAgent(model=self.model, ulog_file=self.ulog_file)
        self.collector = StreamCollector(session_id, user_id)
        self.status = "initialized"
        self.error = None
        
        # Patch the agent to collect information for the API
        self._patch_agent()
    
    def _patch_agent(self):
        """Patch the agent to collect information for the API without modifying the original code."""
        original_process_tool_call = self.agent.process_tool_call
        
        def patched_process_tool_call(tool_call, messages):
            # Debug the tool call structure
            print(f"DEBUG - Original tool_call structure: {json.dumps(tool_call, default=str)}")
            
            # If the tool_call is a dict but doesn't have a 'name' field, try to extract it from the function field
            if isinstance(tool_call, dict):
                if "name" not in tool_call and "function" in tool_call:
                    tool_call["name"] = tool_call["function"].get("name")
                    tool_call["arguments"] = tool_call["function"].get("arguments")
                
                # For topic_data calls specifically
                if tool_call.get("name") == "topic_data" or "topic_data" in str(tool_call):
                    print(f"DEBUG - Detected topic_data tool call")
                    # Ensure it has a name field
                    if "name" not in tool_call:
                        tool_call["name"] = "topic_data"
            
            # Record the tool call
            self.collector.on_tool_call(tool_call)
            
            # Call the original method
            result = original_process_tool_call(tool_call, messages)
            
            # Record the tool result
            if result:
                self.collector.on_tool_result(result)
            
            return result
        
        # Replace the original method with the patched one
        self.agent.process_tool_call = patched_process_tool_call
        
        # Patch the agent's on_thinking method
        original_on_thinking = self.agent.on_thinking
        
        def patched_on_thinking(thinking):
            # Call the original method
            ret = original_on_thinking(thinking)
            
            # Only forward thinking if show_reasoning is True
            if self.show_reasoning:
                self.collector.on_thinking(thinking)
            
            return ret
        
        # Replace the original method with the patched one
        self.agent.on_thinking = patched_on_thinking
        
        # Patch the agent's response processing to capture the final answer
        original_process_response = self.agent.process_response
        
        def patched_process_response(response_text, messages):
            # Call the original method
            ret = original_process_response(response_text, messages)
            
            # Update status
            self.status = "completed"
            
            # Set the final answer
            if response_text:
                self.collector.on_final_answer(response_text)
            
            return ret
        
        # Replace the original method with the patched one
        self.agent.process_response = patched_process_response
    
    def run(self):
        """Run the ULog agent in a separate thread."""
        def run_agent():
            try:
                # Update session status
                session_manager.update_chat_session(self.session_id, {"status": "running"})
                
                # Run the agent
                self.agent.run(self.query)
                
                # Update session status if not already completed
                if self.status != "completed":
                    self.status = "completed"
                    session_manager.update_chat_session(self.session_id, {"status": "completed"})
            except Exception as e:
                # Set error message
                error_message = f"Error running agent: {str(e)}"
                self.error = error_message
                self.status = "error"
                
                # Update session in session manager
                session_manager.set_error(self.session_id, error_message)
                
                print(f"Error running agent: {str(e)}")
        
        # Start the agent in a separate thread
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the session."""
        return self.collector.get_state()
