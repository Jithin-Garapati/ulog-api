"""
API routes for the multi-user ULog Reasoning Agent API.
"""
from flask import request, jsonify, Response, stream_with_context, render_template, send_from_directory
import json
import time
import os
import uuid
from werkzeug.utils import secure_filename

from session_manager import session_manager
from storage import storage
from auth import require_auth, optional_auth
from config import CONFIG

# Import app and core classes from api_multiuser
from api_multiuser import app, ULogAgentSession, allowed_file, UPLOAD_FOLDER

# Root route to serve the index.html file
@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

# Route for client example
@app.route('/client')
def client_example():
    """Serve the client example page."""
    return render_template('client_example.html')

# API Routes
@app.route('/api/auth', methods=['POST'])
def authenticate():
    """
    Authenticate a user.
    Creates a new user session if no credentials are provided.
    Validates existing credentials if provided.
    """
    # Get user_id and session_token from request
    user_id = request.json.get('user_id') if request.json else None
    session_token = request.json.get('session_token') if request.json else None
    
    # If no credentials provided, create a new session
    if not user_id or not session_token:
        user_session = session_manager.create_user_session()
        return jsonify({
            "success": True,
            "message": "New user session created",
            "user_id": user_session["user_id"],
            "session_token": user_session["session_token"]
        })
    
    # Validate existing session
    if session_manager.validate_user_session(user_id, session_token):
        return jsonify({
            "success": True,
            "message": "Session validated",
            "user_id": user_id,
            "session_token": session_token
        })
    else:
        # Create a new session if validation fails
        user_session = session_manager.create_user_session()
        return jsonify({
            "success": True,
            "message": "New user session created (previous session expired)",
            "user_id": user_session["user_id"],
            "session_token": user_session["session_token"]
        })

@app.route('/api/ulog_files', methods=['GET'])
@require_auth
def get_ulog_files(user_id=None):
    """Get a list of available ULog files for the user."""
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
    
    # Get user-specific files from R2 storage
    user_files = storage.get_file_list(user_id)
    ulog_files.extend(user_files)
    
    # Look for ULog files in the uploads directory (for local testing)
    if os.path.exists(UPLOAD_FOLDER):
        for file_path in os.listdir(UPLOAD_FOLDER):
            if file_path.endswith('.ulg'):
                full_path = os.path.join(UPLOAD_FOLDER, file_path)
                file_size = os.path.getsize(full_path)
                file_date = os.path.getmtime(full_path)
                
                # Check if this file is already in the list (from R2)
                if not any(f["name"] == file_path for f in ulog_files):
                    ulog_files.append({
                        "path": full_path,
                        "name": file_path,
                        "size": file_size,
                        "date": file_date,
                        "is_default": False
                    })
    
    return jsonify({
        "files": ulog_files
    })

@app.route('/api/upload_ulog', methods=['POST', 'OPTIONS'])
@require_auth
def upload_ulog(user_id=None):
    """Handle ULog file uploads."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-User-Id, X-Session-Token')
        return response
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        # Upload to Cloudflare R2
        result = storage.upload_file(file, user_id)
        
        if result["success"]:
            # For local development, also save to local uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                file.seek(0)  # Reset file pointer after R2 upload
                file.save(file_path)
            except Exception as e:
                print(f"Warning: Could not save local copy: {str(e)}")
            
            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "file_info": result["file_info"]
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
    else:
        return jsonify({
            "success": False,
            "error": "File type not allowed. Only .ulg files are supported."
        }), 400

@app.route('/api/query', methods=['POST'])
@require_auth
def query(user_id=None):
    """Create a new query session."""
    if not request.json:
        return jsonify({"error": "Missing JSON data"}), 400
    
    query_text = request.json.get('query')
    show_reasoning = request.json.get('show_reasoning', True)
    model = request.json.get('model')
    ulog_file = request.json.get('ulog_file')  # Get ULog file path from request
    
    if not query_text:
        return jsonify({"error": "Missing query parameter"}), 400
    
    # Create a new chat session in the session manager
    chat_session = session_manager.create_chat_session(
        user_id=user_id,
        query=query_text,
        show_reasoning=show_reasoning,
        model=model,
        ulog_file=ulog_file
    )
    
    session_id = chat_session["session_id"]
    
    # Create a ULogAgentSession
    session = ULogAgentSession(
        session_id=session_id, 
        user_id=user_id,
        query=query_text, 
        show_reasoning=show_reasoning,
        model=model,
        ulog_file=ulog_file
    )
    
    # Start the agent
    session.run()
    
    return jsonify({
        "session_id": session_id,
        "status": "running",
        "message": "Query is being processed"
    })

@app.route('/api/sessions/<session_id>', methods=['GET'])
@require_auth
def get_session(session_id, user_id=None):
    """Get the current state of a session."""
    session = session_manager.get_chat_session(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    return jsonify(session)

@app.route('/api/sessions/<session_id>/stream', methods=['GET', 'OPTIONS'])
@require_auth
def stream_session(session_id, user_id=None):
    """Stream updates from a session."""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-User-Id, X-Session-Token')
        return response
        
    session = session_manager.get_chat_session(session_id, user_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Create a new StreamCollector for this session
    collector = ULogAgentSession(
        session_id=session_id,
        user_id=user_id,
        query=session["query"],
        show_reasoning=session["show_reasoning"],
        model=session["model"],
        ulog_file=session["ulog_file"]
    ).collector
    
    def generate():
        """Generate streaming updates."""
        # Send initial state
        yield f"data: {json.dumps(session)}\n\n"
        
        # Generate updates from session
        for update in collector.updates():
            yield f"data: {json.dumps(update)}\n\n"
    
    # Create a streaming response
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )
    
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-User-Id, X-Session-Token')
    response.headers.add('Cache-Control', 'no-cache')
    response.headers.add('Connection', 'keep-alive')
    response.headers.add('X-Accel-Buffering', 'no')  # Disable proxy buffering
    
    return response

@app.route('/api/sessions', methods=['GET'])
@require_auth
def get_user_sessions(user_id=None):
    """Get all sessions for the current user."""
    sessions = session_manager.get_user_chat_sessions(user_id)
    
    return jsonify({
        "sessions": sessions
    })

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@require_auth
def delete_session(session_id, user_id=None):
    """Delete a session."""
    if session_manager.delete_chat_session(session_id, user_id):
        return jsonify({
            "success": True,
            "message": "Session deleted"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to delete session"
        }), 500

@app.route('/api/sessions/clear', methods=['POST'])
@require_auth
def clear_sessions(user_id=None):
    """Clear all sessions for the current user."""
    count = session_manager.clean_up_user_chat_sessions(user_id)
    
    return jsonify({
        "success": True,
        "message": f"Cleared {count} sessions"
    })

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
    # Create a sanitized version of the config that doesn't include sensitive information
    safe_config = {
        "agent": {
            "model": CONFIG["agent"].get("model", "default"),
            "features": CONFIG["agent"].get("features", [])
        },
        "ui": {
            "title": CONFIG.get("ui", {}).get("title", "ULog Reasoning Agent"),
            "description": CONFIG.get("ui", {}).get("description", "Analyze ULog files using natural language")
        }
    }
    
    return jsonify(safe_config)

@app.route('/api/docs', methods=['GET'])
def get_api_docs():
    """Get API documentation."""
    docs = {
        "info": {
            "title": "ULog Reasoning Agent API",
            "version": "1.0.0",
            "description": "Web API for the ULog Reasoning Agent"
        },
        "paths": {
            "/api/auth": {
                "post": {
                    "summary": "Authenticate user",
                    "description": "Create a new user session or validate an existing one",
                    "parameters": [
                        {
                            "name": "user_id",
                            "description": "User ID (optional)",
                            "required": False
                        },
                        {
                            "name": "session_token",
                            "description": "Session token (optional)",
                            "required": False
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Authentication successful",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "success": True,
                                        "message": "New user session created",
                                        "user_id": "user_uuid",
                                        "session_token": "session_uuid"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/ulog_files": {
                "get": {
                    "summary": "Get ULog files",
                    "description": "Get a list of available ULog files for the user",
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "List of ULog files",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "files": [
                                            {
                                                "path": "path/to/file.ulg",
                                                "name": "file.ulg",
                                                "size": 1024,
                                                "date": 1616447402.123,
                                                "is_default": True
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/upload_ulog": {
                "post": {
                    "summary": "Upload ULog file",
                    "description": "Upload a ULog file for analysis",
                    "parameters": [
                        {
                            "name": "file",
                            "description": "ULog file to upload",
                            "required": True
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "File uploaded successfully",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "success": True,
                                        "message": "File uploaded successfully",
                                        "file_info": {
                                            "key": "uploads/user_uuid_timestamp_file.ulg",
                                            "original_name": "file.ulg",
                                            "unique_name": "user_uuid_timestamp_file.ulg",
                                            "size": 1024,
                                            "date": 1616447402.123,
                                            "user_id": "user_uuid"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/query": {
                "post": {
                    "summary": "Create query session",
                    "description": "Create a new query session",
                    "parameters": [
                        {
                            "name": "query",
                            "description": "Query text",
                            "required": True
                        },
                        {
                            "name": "show_reasoning",
                            "description": "Whether to show reasoning (default: true)",
                            "required": False
                        },
                        {
                            "name": "model",
                            "description": "Model to use (default: from config)",
                            "required": False
                        },
                        {
                            "name": "ulog_file",
                            "description": "ULog file to analyze",
                            "required": False
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Query session created",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "session_id": "session_uuid",
                                        "status": "running",
                                        "message": "Query is being processed"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/sessions/{session_id}": {
                "get": {
                    "summary": "Get session",
                    "description": "Get the current state of a session",
                    "parameters": [
                        {
                            "name": "session_id",
                            "description": "Session ID",
                            "required": True
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Session state",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "session_id": "session_uuid",
                                        "user_id": "user_uuid",
                                        "query": "What is the maximum altitude?",
                                        "show_reasoning": True,
                                        "model": "gemini-2.0-flash",
                                        "ulog_file": "file.ulg",
                                        "created_at": 1616447402.123,
                                        "last_updated": 1616447402.123,
                                        "status": "running",
                                        "messages": [],
                                        "final_answer": None,
                                        "error": None
                                    }
                                }
                            }
                        }
                    }
                },
                "delete": {
                    "summary": "Delete session",
                    "description": "Delete a session",
                    "parameters": [
                        {
                            "name": "session_id",
                            "description": "Session ID",
                            "required": True
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Session deleted",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "success": True,
                                        "message": "Session deleted"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/sessions/{session_id}/stream": {
                "get": {
                    "summary": "Stream session updates",
                    "description": "Stream updates from a session",
                    "parameters": [
                        {
                            "name": "session_id",
                            "description": "Session ID",
                            "required": True
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Stream of session updates",
                            "content": {
                                "text/event-stream": {
                                    "example": "data: {...}\n\n"
                                }
                            }
                        }
                    }
                }
            },
            "/api/sessions": {
                "get": {
                    "summary": "Get user sessions",
                    "description": "Get all sessions for the current user",
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "List of sessions",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "sessions": []
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/sessions/clear": {
                "post": {
                    "summary": "Clear sessions",
                    "description": "Clear all sessions for the current user",
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "Sessions cleared",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "success": True,
                                        "message": "Cleared 5 sessions"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/examples": {
                "get": {
                    "summary": "Get examples",
                    "description": "Get example queries",
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "List of examples",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "examples": [
                                            "What was the maximum altitude difference during the flight?"
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/config": {
                "get": {
                    "summary": "Get config",
                    "description": "Get the agent configuration",
                    "parameters": [],
                    "responses": {
                        "200": {
                            "description": "Agent configuration",
                            "content": {
                                "application/json": {
                                    "example": {
                                        "agent": {
                                            "model": "gemini-2.0-flash",
                                            "features": []
                                        },
                                        "ui": {
                                            "title": "ULog Reasoning Agent",
                                            "description": "Analyze ULog files using natural language"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return jsonify(docs)

# Main entry point
if __name__ == '__main__':
    from api_multiuser import app
    app.run(debug=os.getenv('DEBUG', 'False').lower() == 'true',
            host=os.getenv('HOST', '0.0.0.0'),
            port=int(os.getenv('PORT', 5000)))
