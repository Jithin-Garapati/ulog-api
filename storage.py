"""
Cloudflare R2 Storage integration for the ULog Reasoning Agent API.
Handles file uploads, downloads, and session management.
"""
import os
import json
import uuid
import time
import boto3
from typing import Dict, List, Any, Optional, BinaryIO
from dotenv import load_dotenv
from werkzeug.datastructures import FileStorage
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# R2 Storage configuration
R2_ENDPOINT_URL = os.getenv('R2_ENDPOINT_URL')
R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('R2_BUCKET_NAME', 'qwen-agent-data')

# Session configuration
SESSION_EXPIRY_SECONDS = 3600  # 1 hour

class CloudflareR2Storage:
    """Handles storage operations with Cloudflare R2."""
    
    def __init__(self):
        """Initialize the storage client."""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        # Create required folders
        self._ensure_folder_exists('uploads')
        self._ensure_folder_exists('sessions')
    
    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create it if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=BUCKET_NAME)
        except ClientError as e:
            # If bucket doesn't exist, create it
            if e.response['Error']['Code'] == '404':
                self.s3_client.create_bucket(Bucket=BUCKET_NAME)
            else:
                raise
    
    def _ensure_folder_exists(self, folder_name: str):
        """Create a folder (prefix) in the bucket if it doesn't exist."""
        # In S3/R2, folders are just prefixes, we create an empty object with the prefix
        try:
            key = f"{folder_name}/"
            self.s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body='')
        except Exception as e:
            print(f"Error creating folder {folder_name}: {str(e)}")
    
    def upload_file(self, file: FileStorage, user_id: str) -> Dict[str, Any]:
        """
        Upload a file to R2 storage.
        
        Args:
            file: The file to upload
            user_id: The ID of the user uploading the file
            
        Returns:
            Dict with file information
        """
        original_filename = file.filename
        
        # Generate a unique filename to avoid collisions
        timestamp = int(time.time())
        unique_filename = f"{user_id}_{timestamp}_{original_filename}"
        key = f"uploads/{unique_filename}"
        
        try:
            # Upload the file to R2
            self.s3_client.upload_fileobj(file, BUCKET_NAME, key)
            
            # Get file metadata
            response = self.s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
            file_size = response.get('ContentLength', 0)
            file_date = response.get('LastModified', time.time())
            
            return {
                "success": True,
                "message": "File uploaded successfully",
                "file_info": {
                    "key": key,
                    "original_name": original_filename,
                    "unique_name": unique_filename,
                    "size": file_size,
                    "date": file_date.timestamp() if hasattr(file_date, 'timestamp') else time.time(),
                    "user_id": user_id
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error uploading file: {str(e)}"
            }
    
    def download_file(self, key: str) -> Optional[BinaryIO]:
        """
        Download a file from R2 storage.
        
        Args:
            key: The key of the file to download
            
        Returns:
            The file data or None if the file doesn't exist
        """
        try:
            # Create a temporary file
            file_path = f"/tmp/{os.path.basename(key)}"
            
            # Download the file from R2
            self.s3_client.download_file(BUCKET_NAME, key, file_path)
            
            return open(file_path, 'rb')
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return None
    
    def get_file_list(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get list of files for a specific user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            List of file information
        """
        files = []
        
        try:
            # List objects with the user_id prefix
            response = self.s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=f"uploads/{user_id}"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get file metadata
                    key = obj['Key']
                    file_info = self.s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
                    
                    # Extract original filename from the key
                    filename = os.path.basename(key)
                    original_name = '_'.join(filename.split('_')[2:])  # Remove user_id and timestamp
                    
                    files.append({
                        "key": key,
                        "name": original_name,
                        "size": obj.get('Size', 0),
                        "date": obj.get('LastModified').timestamp() if hasattr(obj.get('LastModified'), 'timestamp') else time.time(),
                        "is_default": False
                    })
        except Exception as e:
            print(f"Error getting file list: {str(e)}")
        
        return files
    
    def delete_file(self, key: str) -> bool:
        """
        Delete a file from R2 storage.
        
        Args:
            key: The key of the file to delete
            
        Returns:
            True if the file was deleted, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
            return True
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return False
    
    def save_session(self, session_id: str, user_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save a session to R2 storage.
        
        Args:
            session_id: The ID of the session
            user_id: The ID of the user
            session_data: The session data to save
            
        Returns:
            True if the session was saved, False otherwise
        """
        try:
            # Add metadata to the session
            session_data['user_id'] = user_id
            session_data['last_updated'] = time.time()
            
            # Convert session data to JSON
            session_json = json.dumps(session_data)
            
            # Save the session to R2
            key = f"sessions/{user_id}/{session_id}.json"
            self.s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=key,
                Body=session_json,
                ContentType='application/json'
            )
            
            return True
        except Exception as e:
            print(f"Error saving session: {str(e)}")
            return False
    
    def get_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a session from R2 storage.
        
        Args:
            session_id: The ID of the session
            user_id: The ID of the user
            
        Returns:
            The session data or None if the session doesn't exist
        """
        try:
            # Get the session from R2
            key = f"sessions/{user_id}/{session_id}.json"
            response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
            
            # Parse the session data
            session_data = json.loads(response['Body'].read().decode('utf-8'))
            
            return session_data
        except Exception as e:
            print(f"Error getting session: {str(e)}")
            return None
    
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a session from R2 storage.
        
        Args:
            session_id: The ID of the session
            user_id: The ID of the user
            
        Returns:
            True if the session was deleted, False otherwise
        """
        try:
            # Delete the session from R2
            key = f"sessions/{user_id}/{session_id}.json"
            self.s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
            
            return True
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
            return False
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            List of session data
        """
        sessions = []
        
        try:
            # List objects with the user_id prefix
            response = self.s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=f"sessions/{user_id}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get session data
                    key = obj['Key']
                    session_response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
                    session_data = json.loads(session_response['Body'].read().decode('utf-8'))
                    
                    sessions.append(session_data)
        except Exception as e:
            print(f"Error getting user sessions: {str(e)}")
        
        return sessions
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions and their associated files.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        current_time = time.time()
        
        try:
            # List all sessions
            response = self.s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix="sessions/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get session data
                    key = obj['Key']
                    try:
                        session_response = self.s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
                        session_data = json.loads(session_response['Body'].read().decode('utf-8'))
                        
                        # Check if the session is expired
                        last_updated = session_data.get('last_updated', 0)
                        if current_time - last_updated > SESSION_EXPIRY_SECONDS:
                            # Get user_id and session_id from the key
                            parts = key.split('/')
                            if len(parts) >= 3:
                                user_id = parts[1]
                                session_id = parts[2].split('.')[0]
                                
                                # Delete the session
                                self.delete_session(session_id, user_id)
                                
                                # Delete associated files
                                self._cleanup_session_files(session_id)
                                
                                cleaned_count += 1
                    except Exception as e:
                        print(f"Error processing session {key}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning up expired sessions: {str(e)}")
        
        return cleaned_count
    
    def _cleanup_session_files(self, session_id: str) -> int:
        """
        Clean up files associated with a session.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        
        try:
            # List objects with session_id in metadata
            response = self.s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix="uploads/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    try:
                        # Get file metadata
                        metadata = self.s3_client.head_object(Bucket=BUCKET_NAME, Key=key).get('Metadata', {})
                        
                        # Check if this file is associated with the session
                        if metadata.get('session_id') == session_id:
                            # Delete the file
                            self.s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
                            cleaned_count += 1
                    except Exception as e:
                        print(f"Error processing file {key}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning up session files: {str(e)}")
        
        return cleaned_count

# Create a singleton instance
storage = CloudflareR2Storage()
