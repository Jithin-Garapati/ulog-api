"""
Session management for the ULog Reasoning Agent API.
Handles user sessions, authentication, and cleanup.
"""
import os
import uuid
import time
import json
import threading
from typing import Dict, List, Any, Optional
from werkzeug.utils import secure_filename

from storage import storage

# Session configuration
SESSION_CLEANUP_INTERVAL = 3600  # Clean up expired sessions every hour
USER_SESSION_EXPIRY = 86400  # User sessions expire after 24 hours

class SessionManager:
    """Manages user sessions and authentication."""
    
    def __init__(self):
        """Initialize the session manager."""
        self.active_sessions = {}  # In-memory cache of active sessions
        self.active_users = {}  # Map user_id to their active sessions
        self.session_locks = {}  # Locks for thread-safe session access
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        self.cleanup_thread.start()
    
    def create_user_session(self) -> Dict[str, Any]:
        """
        Create a new user session.
        
        Returns:
            Dict with user session information
        """
        user_id = str(uuid.uuid4())
        session_token = str(uuid.uuid4())
        
        user_session = {
            "user_id": user_id,
            "session_token": session_token,
            "created_at": time.time(),
            "last_active": time.time(),
            "chat_sessions": []  # List of chat session IDs
        }
        
        # Store the user session
        self.active_users[user_id] = user_session
        
        return user_session
    
    def validate_user_session(self, user_id: str, session_token: str) -> bool:
        """
        Validate a user session.
        
        Args:
            user_id: The ID of the user
            session_token: The session token
            
        Returns:
            True if the session is valid, False otherwise
        """
        if user_id not in self.active_users:
            return False
        
        user_session = self.active_users[user_id]
        
        # Check if the session token matches
        if user_session["session_token"] != session_token:
            return False
        
        # Check if the session has expired
        if time.time() - user_session["last_active"] > USER_SESSION_EXPIRY:
            # Remove the expired session
            del self.active_users[user_id]
            return False
        
        # Update last active time
        user_session["last_active"] = time.time()
        
        return True
    
    def create_chat_session(self, user_id: str, query: str, show_reasoning: bool = True, 
                          model: str = None, ulog_file: str = None) -> Dict[str, Any]:
        """
        Create a new chat session.
        
        Args:
            user_id: The ID of the user
            query: The query text
            show_reasoning: Whether to show reasoning
            model: The model to use
            ulog_file: The ULog file to use
            
        Returns:
            Dict with chat session information
        """
        if user_id not in self.active_users:
            return {"error": "Invalid user session"}
        
        # Generate a new session ID
        session_id = str(uuid.uuid4())
        
        # Create a lock for this session
        self.session_locks[session_id] = threading.Lock()
        
        # Create the chat session
        chat_session = {
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "show_reasoning": show_reasoning,
            "model": model,
            "ulog_file": ulog_file,
            "created_at": time.time(),
            "last_updated": time.time(),
            "status": "initialized",
            "messages": [],
            "final_answer": None,
            "error": None
        }
        
        # Add the session to the user's chat sessions
        user_session = self.active_users[user_id]
        user_session["chat_sessions"].append(session_id)
        
        # Store the chat session
        self.active_sessions[session_id] = chat_session
        
        # Save to persistent storage
        storage.save_session(session_id, user_id, chat_session)
        
        return chat_session
    
    def get_chat_session(self, session_id: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get a chat session.
        
        Args:
            session_id: The ID of the chat session
            user_id: The ID of the user (optional validation)
            
        Returns:
            The chat session or None if not found
        """
        # Check in-memory cache first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Validate user_id if provided
            if user_id is not None and session["user_id"] != user_id:
                return None
            
            return session
        
        # If not in memory, try to load from storage
        session = storage.get_session(session_id, user_id if user_id else "unknown")
        if session:
            # Cache the session in memory
            self.active_sessions[session_id] = session
            return session
        
        return None
    
    def update_chat_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a chat session.
        
        Args:
            session_id: The ID of the chat session
            updates: The updates to apply
            
        Returns:
            True if the session was updated, False otherwise
        """
        session = self.get_chat_session(session_id)
        if not session:
            return False
        
        # Acquire the lock for this session
        with self.session_locks.get(session_id, threading.Lock()):
            # Update the session
            session.update(updates)
            session["last_updated"] = time.time()
            
            # Update in-memory cache
            self.active_sessions[session_id] = session
            
            # Save to persistent storage
            return storage.save_session(session_id, session["user_id"], session)
    
    def add_message_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Add a message to a chat session.
        
        Args:
            session_id: The ID of the chat session
            message: The message to add
            
        Returns:
            True if the message was added, False otherwise
        """
        session = self.get_chat_session(session_id)
        if not session:
            return False
        
        # Add unique ID and timestamp if not present
        if "id" not in message:
            message["id"] = str(uuid.uuid4())
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        
        # Acquire the lock for this session
        with self.session_locks.get(session_id, threading.Lock()):
            # Add the message
            session["messages"].append(message)
            session["last_updated"] = time.time()
            
            # Update in-memory cache
            self.active_sessions[session_id] = session
            
            # Save to persistent storage
            return storage.save_session(session_id, session["user_id"], session)
    
    def set_final_answer(self, session_id: str, final_answer: str) -> bool:
        """
        Set the final answer for a chat session.
        
        Args:
            session_id: The ID of the chat session
            final_answer: The final answer
            
        Returns:
            True if the final answer was set, False otherwise
        """
        session = self.get_chat_session(session_id)
        if not session:
            return False
        
        # Acquire the lock for this session
        with self.session_locks.get(session_id, threading.Lock()):
            # Set the final answer
            session["final_answer"] = final_answer
            session["status"] = "completed"
            session["last_updated"] = time.time()
            
            # Update in-memory cache
            self.active_sessions[session_id] = session
            
            # Save to persistent storage
            return storage.save_session(session_id, session["user_id"], session)
    
    def set_error(self, session_id: str, error: str) -> bool:
        """
        Set an error for a chat session.
        
        Args:
            session_id: The ID of the chat session
            error: The error message
            
        Returns:
            True if the error was set, False otherwise
        """
        session = self.get_chat_session(session_id)
        if not session:
            return False
        
        # Acquire the lock for this session
        with self.session_locks.get(session_id, threading.Lock()):
            # Set the error
            session["error"] = error
            session["status"] = "error"
            session["last_updated"] = time.time()
            
            # Update in-memory cache
            self.active_sessions[session_id] = session
            
            # Save to persistent storage
            return storage.save_session(session_id, session["user_id"], session)
    
    def delete_chat_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: The ID of the chat session
            user_id: The ID of the user
            
        Returns:
            True if the session was deleted, False otherwise
        """
        session = self.get_chat_session(session_id, user_id)
        if not session:
            return False
        
        # Remove from in-memory cache
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Remove from user's chat sessions
        if user_id in self.active_users:
            user_session = self.active_users[user_id]
            if session_id in user_session["chat_sessions"]:
                user_session["chat_sessions"].remove(session_id)
        
        # Remove the lock
        if session_id in self.session_locks:
            del self.session_locks[session_id]
        
        # Remove from persistent storage
        return storage.delete_session(session_id, user_id)
    
    def get_user_chat_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            List of chat sessions
        """
        if user_id not in self.active_users:
            return []
        
        user_session = self.active_users[user_id]
        sessions = []
        
        for session_id in user_session["chat_sessions"]:
            session = self.get_chat_session(session_id, user_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    def clean_up_user_chat_sessions(self, user_id: str) -> int:
        """
        Clean up all chat sessions for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Number of sessions cleaned up
        """
        if user_id not in self.active_users:
            return 0
        
        user_session = self.active_users[user_id]
        count = 0
        
        for session_id in list(user_session["chat_sessions"]):
            if self.delete_chat_session(session_id, user_id):
                count += 1
        
        return count
    
    def _cleanup_thread(self):
        """Thread to periodically clean up expired sessions."""
        while True:
            try:
                # Sleep for the cleanup interval
                time.sleep(SESSION_CLEANUP_INTERVAL)
                
                # Clean up expired sessions from storage
                storage.cleanup_expired_sessions()
                
                # Clean up expired user sessions
                for user_id in list(self.active_users.keys()):
                    user_session = self.active_users[user_id]
                    if time.time() - user_session["last_active"] > USER_SESSION_EXPIRY:
                        self.clean_up_user_chat_sessions(user_id)
                        del self.active_users[user_id]
                
                # Clean up expired chat sessions from memory
                for session_id in list(self.active_sessions.keys()):
                    session = self.active_sessions[session_id]
                    if time.time() - session["last_updated"] > USER_SESSION_EXPIRY:
                        if session_id in self.session_locks:
                            del self.session_locks[session_id]
                        del self.active_sessions[session_id]
            except Exception as e:
                print(f"Error in cleanup thread: {str(e)}")

# Create a singleton instance
session_manager = SessionManager()
