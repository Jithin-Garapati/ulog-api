"""
API server for the ULog Reasoning Agent.
This provides a web API for the ULog Reasoning Agent without modifying existing code.
"""
from flask import Flask, request, jsonify, Response, stream_with_context, render_template, send_from_directory
from flask_cors import CORS
import json
from typing import Dict, List, Any, Optional
import uuid
import time
import threading
import os
import glob
from werkzeug.utils import secure_filename

from agent import ULogReasoningAgent
from config import CONFIG

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Store active sessions
active_sessions = {}

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
    
    def __init__(self, session_id: str):
        self.session_id = session_id
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
        
        self.messages.append({
            "type": "tool_call",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "tool_name": tool_name,
            "args": tool_args
        })
    
    def on_tool_result(self, tool_result: Dict):
        """Handle result of a tool call."""
        # Note: We use the tool_result data structure from the agent directly
        self.messages.append({
            "type": "tool_result",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "result": tool_result.get("content", "")
        })
    
    def on_final_answer(self, final_answer: str):
        """Handle the final answer from the agent."""
        self.flush_current_message()
        self.final_answer = final_answer
        self.messages.append({
            "type": "final_answer",
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "content": final_answer
        })
        
    def updates(self):
        """Generate updates from the collector."""
        # Yield any existing messages first
        for message in self.messages:
            yield message
        
        # Keep yielding new messages as they come in
        last_message_index = len(self.messages)
        while True:
            # Check if there are new messages
            if len(self.messages) > last_message_index:
                # Yield new messages
                for i in range(last_message_index, len(self.messages)):
                    yield self.messages[i]
                last_message_index = len(self.messages)
            
            # Check if session is done
            if self.final_answer is not None:
                # Send completion message
                yield {
                    "type": "complete",
                    "final_answer": self.final_answer
                }
                break
            
            # Small sleep to avoid tight loop
            time.sleep(0.1)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the session."""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "final_answer": self.final_answer
        }

class ULogAgentSession:
    """Manages a session with the ULog agent."""
    
    def __init__(self, session_id: str, query: str, show_reasoning: bool = True, model: str = None, ulog_file: str = None):
        """Initialize a new session."""
        self.session_id = session_id
        self.query = query
        self.show_reasoning = show_reasoning
        self.model = model or CONFIG["agent"]["model"]
        self.ulog_file = ulog_file  # Store the ULog file path
        self.agent = ULogReasoningAgent(model=self.model, ulog_file=self.ulog_file)
        self.collector = StreamCollector(session_id)
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
            if len(messages) > 0 and messages[-1]["role"] == "tool":
                self.collector.on_tool_result(messages[-1])
            
            return result
        
        # Replace the method
        self.agent.process_tool_call = patched_process_tool_call
        
        # Store the original run method
        self.original_run = self.agent.run
        
        # Hook into the Gemini client if needed
        if self.agent.api_provider == "gemini":
            # Store the original stream processing function
            original_stream_process = self.agent.client.chat_completions_create
            
            # Create a patched version to capture the streaming content
            def patched_chat_completions_create(*args, **kwargs):
                # Only patch streaming requests
                if kwargs.get('stream', False):
                    original_stream = original_stream_process(*args, **kwargs)
                    
                    # Return a wrapped generator that captures the content
                    def stream_wrapper():
                        try:
                            for chunk in original_stream:
                                try:
                                    # Extract content from delta if available
                                    delta = chunk.choices[0].delta
                                    if hasattr(delta, 'content') and delta.content:
                                        # Send to our collector
                                        self.collector.on_thinking(delta.content)
                                except Exception as e:
                                    # If there's an error processing a chunk, log it but don't break the stream
                                    print(f"Warning: Error processing stream chunk: {str(e)}")
                                
                                # Pass the chunk through
                                yield chunk
                        except Exception as e:
                            # If the stream itself breaks, log it
                            print(f"Error in stream wrapper: {str(e)}")
                            # Don't re-raise, just end the stream
                    
                    return stream_wrapper()
                else:
                    # Non-streaming requests pass through unchanged
                    return original_stream_process(*args, **kwargs)
            
            # Replace the method
            self.agent.client.chat_completions_create = patched_chat_completions_create
    
    def run(self):
        """Run the agent in a separate thread."""
        def run_agent():
            try:
                self.status = "running"
                
                # Override the run method to capture output
                def patched_run(query, show_reasoning=True):
                    try:
                        # Call the original method
                        result = self.original_run(query, show_reasoning)
                        
                        # Record the final answer
                        self.collector.on_final_answer(result)
                        self.status = "completed"
                        
                        return result
                    except Exception as e:
                        self.status = "error"
                        self.error = str(e)
                        error_message = f"Error: {str(e)}"
                        self.collector.on_final_answer(error_message)
                        return error_message
                
                # Replace the run method
                self.agent.run = patched_run
                
                # Start the agent
                result = self.agent.run(self.query, self.show_reasoning)
                return result
            except Exception as e:
                self.status = "error"
                self.error = str(e)
                self.collector.on_final_answer(f"Error: {str(e)}")
        
        # Start the agent in a separate thread
        thread = threading.Thread(target=run_agent)
        thread.daemon = True
        thread.start()
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the session."""
        state = self.collector.get_state()
        state["status"] = self.status
        state["query"] = self.query
        state["model"] = self.model
        state["show_reasoning"] = self.show_reasoning
        state["ulog_file"] = self.ulog_file
        if self.error:
            state["error"] = self.error
        return state

# API routes

# Root route to serve the index.html file
@app.route('/', methods=['GET'])
def index():
    """Serve the main application page."""
    return send_from_directory('.', 'client_example.html')

# Route for client example
@app.route('/client_example.html', methods=['GET'])
def client_example():
    """Serve the client example page."""
    return send_from_directory('.', 'client_example.html')

@app.route('/api/ulog_files', methods=['GET'])
def get_ulog_files():
    """Get a list of available ULog files."""
    # Look for ULog files in the current directory and its subdirectories
    ulog_files = []
    
    # Check the default ULog file from config
    default_ulog = CONFIG["files"].get("ulog_file", "flight_log.ulg")
    if os.path.exists(default_ulog):
        file_size = os.path.getsize(default_ulog)
        file_date = os.path.getmtime(default_ulog)
        ulog_files.append({
            "path": default_ulog,
            "name": os.path.basename(default_ulog),
            "size": file_size,
            "date": file_date,
            "is_default": True
        })
    
    # Look for ULog files in the uploads directory
    if os.path.exists(UPLOAD_FOLDER):
        for file_path in glob.glob(f"{UPLOAD_FOLDER}/*.ulg"):
            # Get file information
            file_path = os.path.normpath(file_path)
            file_size = os.path.getsize(file_path)
            file_date = os.path.getmtime(file_path)
            
            ulog_files.append({
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": file_size,
                "date": file_date,
                "is_default": False
            })
    
    # Also check for other ULog files in the main directory
    for file_path in glob.glob("*.ulg"):
        # Skip if it's the default file (already added)
        if file_path == default_ulog:
            continue
            
        # Get file information
        file_path = os.path.normpath(file_path)
        file_size = os.path.getsize(file_path)
        file_date = os.path.getmtime(file_path)
        
        ulog_files.append({
            "path": file_path,
            "name": os.path.basename(file_path),
            "size": file_size,
            "date": file_date,
            "is_default": False
        })
    
    return jsonify({
        "files": ulog_files
    })

@app.route('/api/upload_ulog', methods=['POST', 'OPTIONS'])
def upload_ulog():
    """Handle ULog file uploads."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        
        # Save the file to the uploads directory
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            file.save(file_path)
            
            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "file_path": file_path
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error saving file: {str(e)}"
            }), 500
    else:
        return jsonify({
            "success": False,
            "error": "File type not allowed. Only .ulg files are supported."
        }), 400

@app.route('/api/query', methods=['POST'])
def query():
    """Create a new query session."""
    if not request.json:
        return jsonify({"error": "Missing JSON data"}), 400
    
    query_text = request.json.get('query')
    show_reasoning = request.json.get('show_reasoning', True)
    model = request.json.get('model')
    ulog_file = request.json.get('ulog_file')  # Get ULog file path from request
    
    if not query_text:
        return jsonify({"error": "Missing query parameter"}), 400
    
    session_id = str(uuid.uuid4())
    
    # Create a new session
    session = ULogAgentSession(
        session_id=session_id, 
        query=query_text, 
        show_reasoning=show_reasoning,
        model=model,
        ulog_file=ulog_file  # Pass ULog file path to session
    )
    active_sessions[session_id] = session
    
    # Start the agent
    session.run()
    
    return jsonify({
        "session_id": session_id,
        "status": "running",
        "message": "Query is being processed"
    })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get the current state of a session."""
    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = active_sessions[session_id]
    return jsonify(session.get_state())

@app.route('/api/sessions/<session_id>/stream', methods=['GET', 'OPTIONS'])
def stream_session(session_id):
    """Stream updates from a session."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
        
    if session_id not in active_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    session = active_sessions[session_id]
    
    def generate():
        """Generate streaming updates."""
        # Send initial state
        state = session.get_state()
        yield f"data: {json.dumps(state)}\n\n"
        
        # Generate updates from session
        for update in session.collector.updates():
            yield f"data: {json.dumps(update)}\n\n"
    
    # Create a streaming response
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
    
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')  # Disable proxy buffering
    
    return response

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example queries."""
    example_queries = [
        "What was the maximum altitude difference during the flight?",
        "What is the maximum tilt angle during the flight?",
        "what was the maximum speed during the flight?",
        "How steady was the aircraft's movement during the flight?",
        "Using estimated position information, find the deviation of flight path from the setpoint.",
        "Extract the vz field from vehicle_local_position and compute all timestamps where vertical velocity is negative (vz < 0), which indicates descent."
    ]
    
    return jsonify({
        "examples": example_queries
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the agent configuration."""
    # Filter sensitive information
    safe_config = {
        "agent": {
            "model": CONFIG["agent"]["model"],
            "api_provider": CONFIG["agent"].get("api_provider", "groq"),
            "temperature": CONFIG["agent"].get("temperature", 0.1),
            "max_tokens": CONFIG["agent"].get("max_tokens", 1024)
        }
    }
    
    return jsonify(safe_config)

# API documentation endpoint
@app.route('/api/docs', methods=['GET'])
def get_api_docs():
    """Get API documentation."""
    docs = {
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/api/ulog_files",
                "method": "GET",
                "description": "Get a list of available ULog files",
                "returns": {
                    "files": "Array - List of ULog files"
                }
            },
            {
                "path": "/api/upload_ulog",
                "method": "POST",
                "description": "Upload a ULog file",
                "parameters": {
                    "file": "File - The ULog file to upload"
                },
                "returns": {
                    "success": "Boolean - Whether the upload was successful",
                    "message": "String - A message describing the result",
                    "file_path": "String - The path to the uploaded file"
                }
            },
            {
                "path": "/api/query",
                "method": "POST",
                "description": "Create a new query session",
                "parameters": {
                    "query": "String - The query to process",
                    "show_reasoning": "Boolean (optional) - Whether to include reasoning in the response",
                    "model": "String (optional) - The model to use for processing",
                    "ulog_file": "String (optional) - The ULog file to use for processing"
                },
                "returns": {
                    "session_id": "String - The ID of the created session",
                    "status": "String - The status of the session",
                    "message": "String - A message describing the status"
                }
            },
            {
                "path": "/api/sessions/<session_id>",
                "method": "GET",
                "description": "Get the current state of a session",
                "parameters": {
                    "session_id": "String - The ID of the session to get"
                },
                "returns": {
                    "session_id": "String - The ID of the session",
                    "messages": "Array - The messages in the session",
                    "final_answer": "String - The final answer of the session",
                    "status": "String - The status of the session",
                    "query": "String - The query that was processed",
                    "model": "String - The model that was used",
                    "show_reasoning": "Boolean - Whether reasoning was included",
                    "ulog_file": "String - The ULog file that was used"
                }
            },
            {
                "path": "/api/sessions/<session_id>/stream",
                "method": "GET",
                "description": "Stream the current state of a session",
                "parameters": {
                    "session_id": "String - The ID of the session to stream"
                },
                "returns": "Stream of JSON objects, each representing a message in the session"
            },
            {
                "path": "/api/examples",
                "method": "GET",
                "description": "Get example queries",
                "returns": {
                    "examples": "Array - Example queries"
                }
            },
            {
                "path": "/api/config",
                "method": "GET",
                "description": "Get the agent configuration",
                "returns": {
                    "agent": "Object - The agent configuration"
                }
            }
        ],
        "message_types": [
            {
                "type": "thinking",
                "description": "Reasoning from the agent"
            },
            {
                "type": "tool_call",
                "description": "A tool call from the agent"
            },
            {
                "type": "tool_result",
                "description": "A tool result from a tool call"
            },
            {
                "type": "final_answer",
                "description": "The final answer from the agent"
            }
        ]
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
