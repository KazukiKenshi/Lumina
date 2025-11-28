from flask import Flask, request, jsonify
from flask_cors import CORS
from src.main import ChatSession
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Store active sessions in memory (use Redis/database for production)
sessions = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Mental Health Chatbot API is running"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint
    Expected JSON body:
    {
        "message": "user message text",
        "session_id": "optional_session_id",
        "user_id": "optional_user_id"
    }
    
    Returns:
    {
        "text": "response text",
        "emotion": "neutral|sad|happy",
        "session_id": "session_id",
        "timestamp": "ISO timestamp"lumina-chat  | [DB DEBUG] get_recent_exchanges called with n=4, user_id=hiteshd258@gmail.com
lumina-chat  | [DB ERROR] Could not connect to MongoDB: localhost:27017: [Errno 111] Connection refused (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms), Timeout: 3.0s, Topology Description: <TopologyDescription id: 6924f0a44bcdb35b4fc902ed, topology_type: Unknown, servers: [<ServerDescription ('localhost', 27017) server_type: Unknown, rtt: None, error=AutoReconnect('localhost:27017: [Errno 111] Connection refused (configured timeouts: socketTimeoutMS: 20000.0ms, connectTimeoutMS: 20000.0ms)')>]>
lumina-chat  | [API] Created new session: 20251124_235623 for user: hiteshd258@gmail.com
lumina-chat  | [API] Processing message from user: hiteshd258@gmail.com
lumina-chat  | [API] Message: [neutral face] hello
lumina-chat  | [MODE] WELLNESS (confidence: 0.19)
lumina-chat  | [RETRIEVE] mode=wellness k=3 tags=[]
lumina-chat  | [ERROR] Mistral API error: API error occurred: Status 401 Content-Type "application/json; charset=utf-8". Body: {"detail":"Unauthorized"}
lumina-chat  | 
lumina-chat  | [2025-11-24 23:56:25] Mode: wellness | Risk: N/A
lumina-chat  | User: [neutral face] hello
lumina-chat  | 23:56:25 INFO: -----
lumina-chat  | Session: 20251124_235623
lumina-chat  | Timestamp: 2025-11-24 23:56:25
lumina-chat  | Mode: wellness | Risk: N/A
lumina-chat  | User: [neutral face] hello
lumina-chat  | Assistant: I apologize, but I'm having technical difficulties right now. Please try again in a moment. If you're in crisis, please contact emergency services or a crisis helpline immediately. Error: 'NoneType' object has no attribute 'choices'
lumina-chat  | 172.19.0.3 - - [24/Nov/2025 23:56:25] "POST /api/chat HTTP/1.1" 200 -
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request body"
            }), 400
        
        user_message = data['message']
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'default')
        
        # Get or create session
        if session_id and session_id in sessions:
            session = sessions[session_id]
            print(f"[API] Using existing session: {session_id}")
            session.set_user(user_id)  # Refresh session if user changes
        else:
            session = ChatSession(user_id=user_id)
            session_id = session.state['session_id']
            sessions[session_id] = session
            print(f"[API] Created new session: {session_id} for user: {user_id}")
        
        # Process message
        print(f"[API] Processing message from user: {user_id}")
        print(f"[API] Message: {user_message}")
        
        response = session.chat(user_message)
        
        # Return response with metadata
        return jsonify({
            "text": response['text'],
            "emotion": response['emotion'],
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": session.state['conversation_history'][-1]['timestamp']
        })
        
    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/api/session/<session_id>/summary', methods=['GET'])
def get_session_summary(session_id):
    """Get summary of a session"""
    if session_id not in sessions:
        return jsonify({
            "error": "Session not found"
        }), 404
    
    session = sessions[session_id]
    summary = session.get_summary()
    
    return jsonify(summary)

@app.route('/api/session/<session_id>/history', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history for a session"""
    if session_id not in sessions:
        return jsonify({
            "error": "Session not found"
        }), 404
    
    session = sessions[session_id]
    history = session.state.get('conversation_history', [])
    
    return jsonify({
        "session_id": session_id,
        "conversation_history": history
    })

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return jsonify({
            "message": "Session deleted successfully",
            "session_id": session_id
        })
    else:
        return jsonify({
            "error": "Session not found"
        }), 404

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    return jsonify({
        "active_sessions": list(sessions.keys()),
        "count": len(sessions)
    })

if __name__ == '__main__':
    port = int(os.getenv('CHATBOT_PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("Mental Health Chatbot API Server")
    print(f"Starting on port {port}")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health - Health check")
    print("  POST /api/chat - Send message and get response")
    print("  GET  /api/session/<id>/summary - Get session summary")
    print("  GET  /api/session/<id>/history - Get conversation history")
    print("  DELETE /api/session/<id> - Delete session")
    print("  GET  /api/sessions - List all active sessions")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
