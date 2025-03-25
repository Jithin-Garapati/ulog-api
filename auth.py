"""
Authentication middleware for the ULog Reasoning Agent API.
Handles user authentication and session validation.
"""
from functools import wraps
from flask import request, jsonify
from session_manager import session_manager

def require_auth(f):
    """
    Decorator to require authentication for API routes.
    Checks for user_id and session_token in the request.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get authentication from headers
        user_id = request.headers.get("X-User-Id")
        session_token = request.headers.get("X-Session-Token")
        
        # Get authentication from query parameters (fallback)
        if not user_id:
            user_id = request.args.get("user_id")
        if not session_token:
            session_token = request.args.get("session_token")
        
        # If no authentication provided, create a new session
        if not user_id or not session_token:
            # Create a new user session
            user_session = session_manager.create_user_session()
            
            # Return the new session info
            response = jsonify({
                "message": "New user session created",
                "user_id": user_session["user_id"],
                "session_token": user_session["session_token"]
            })
            
            # Set authentication headers
            response.headers["X-User-Id"] = user_session["user_id"]
            response.headers["X-Session-Token"] = user_session["session_token"]
            
            return response
        
        # Validate the session
        if not session_manager.validate_user_session(user_id, session_token):
            return jsonify({"error": "Invalid or expired session"}), 401
        
        # Set the user_id in kwargs to pass to the route handler
        kwargs["user_id"] = user_id
        
        return f(*args, **kwargs)
    
    return decorated

def optional_auth(f):
    """
    Decorator to optionally authenticate for API routes.
    If authentication is provided, validates it.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Get authentication from headers
        user_id = request.headers.get("X-User-Id")
        session_token = request.headers.get("X-Session-Token")
        
        # Get authentication from query parameters (fallback)
        if not user_id:
            user_id = request.args.get("user_id")
        if not session_token:
            session_token = request.args.get("session_token")
        
        # If authentication provided, validate it
        if user_id and session_token:
            if session_manager.validate_user_session(user_id, session_token):
                # Set the user_id in kwargs to pass to the route handler
                kwargs["user_id"] = user_id
            else:
                # Don't return an error, just don't set user_id
                pass
        
        return f(*args, **kwargs)
    
    return decorated
