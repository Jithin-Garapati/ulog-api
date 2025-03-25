import os
import uuid
import json
import time
import shutil
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
from flask_cors import CORS
import threading
import tempfile

from agent import ULogReasoningAgent
from config import CONFIG

app = Flask(__name__)
CORS(app)

# Session storage - this will store all active sessions
# Format: {session_id: {agent: ULogReasoningAgent, last_active: timestamp, 
#                       temp_dir: path, ulog_file: filename}}
active_sessions = {}

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure session management
SESSION_TIMEOUT = 60 * 30  # 30 minutes in seconds
CLEANUP_INTERVAL = 60 * 5  # Run cleanup every 5 minutes

# Lock for thread-safe operations on active_sessions
sessions_lock = threading.RLock()

def create_user_temp_dir():
    """Create a temporary directory for a user session"""
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    return temp_dir

def cleanup_expired_sessions():
    """Remove expired sessions and their temporary files"""
    current_time = time.time()
    with sessions_lock:
        expired_sessions = []
        for session_id, session_data in active_sessions.items():
            if current_time - session_data['last_active'] > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            cleanup_session(session_id)
            
def cleanup_session(session_id):
    """Clean up a specific session and its resources"""
    with sessions_lock:
        if session_id in active_sessions:
            # Remove temporary directory and all its contents
            try:
                if 'temp_dir' in active_sessions[session_id]:
                    temp_dir = active_sessions[session_id]['temp_dir']
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up session {session_id}: {e}")
            
            # Remove from active sessions
            del active_sessions[session_id]
            print(f"Cleaned up session {session_id}")

# Start the cleanup thread
def run_periodic_cleanup():
    """Run the cleanup task periodically"""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        cleanup_expired_sessions()

# Start the cleanup thread
cleanup_thread = threading.Thread(target=run_periodic_cleanup, daemon=True)
cleanup_thread.start()

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    temp_dir = create_user_temp_dir()
    
    with sessions_lock:
        active_sessions[session_id] = {
            'agent': None,
            'last_active': time.time(),
            'temp_dir': temp_dir,
            'ulog_file': None
        }
    
    return jsonify({
        'success': True,
        'session_id': session_id
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a ULog file"""
    session_id = request.form.get('session_id')
    
    if not session_id or session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if file:
        # Get the user's temp directory
        with sessions_lock:
            if session_id not in active_sessions:
                return jsonify({'success': False, 'error': 'Session expired'}), 400
                
            temp_dir = active_sessions[session_id]['temp_dir']
            active_sessions[session_id]['last_active'] = time.time()
        
        # Save the file to the temp directory
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Create a new agent for this session
        model_name = request.form.get('model', CONFIG.get('agent', {}).get('default_model', 'qwen-qwq-32b'))
        agent = ULogReasoningAgent(model=model_name, ulog_file=filepath)
        
        # Update session data
        with sessions_lock:
            active_sessions[session_id]['agent'] = agent
            active_sessions[session_id]['ulog_file'] = filepath
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': filename
        })

@app.route('/api/query', methods=['POST'])
def query():
    """Process a query using the agent"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in active_sessions:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    
    query_text = data.get('query')
    if not query_text:
        return jsonify({'success': False, 'error': 'No query provided'}), 400
    
    # Get the agent from the session
    with sessions_lock:
        if session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Session expired'}), 400
            
        session_data = active_sessions[session_id]
        agent = session_data.get('agent')
        session_data['last_active'] = time.time()
    
    if not agent:
        return jsonify({'success': False, 'error': 'No ULog file uploaded for this session'}), 400
    
    # If streaming is requested
    if data.get('stream', False):
        def generate():
            """Generate streaming response"""
            try:
                response = agent.run(query_text, show_reasoning=True)
                yield json.dumps({'type': 'final', 'content': response}) + '\n'
            except Exception as e:
                yield json.dumps({'type': 'error', 'content': str(e)}) + '\n'
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
    
    # Non-streaming response
    try:
        response = agent.run(query_text, show_reasoning=True)
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/session', methods=['DELETE'])
def end_session():
    """End a user session and clean up resources"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'success': False, 'error': 'No session ID provided'}), 400
    
    cleanup_session(session_id)
    return jsonify({
        'success': True, 
        'message': 'Session ended successfully'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway deployment"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'timestamp': time.time()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)