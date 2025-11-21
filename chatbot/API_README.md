# Mental Health Chatbot API

A Flask-based REST API for the mental health chatbot with emotion detection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (copy `.env.example` to `.env` and fill in):
```bash
MISTRAL_API_KEY=your_key_here
CHATBOT_PORT=5001
```

3. Run the API server:
```bash
python api_server.py
```

## API Endpoints

### POST /api/chat
Send a message and get a response with emotion.

**Request:**
```json
{
  "message": "I've been feeling really sad lately",
  "session_id": "optional_session_id",
  "user_id": "optional_user_id"
}
```

**Response:**
```json
{
  "text": "I hear you, and I'm sorry you're feeling this way...",
  "emotion": "sad",
  "session_id": "20250121_143022",
  "user_id": "user123",
  "timestamp": "2025-01-21T14:30:22.123456"
}
```

### GET /health
Health check endpoint.

### GET /api/session/{session_id}/summary
Get session summary (message count, diagnosed conditions, etc).

### GET /api/session/{session_id}/history
Get full conversation history for a session.

### DELETE /api/session/{session_id}
Delete a session and free up memory.

### GET /api/sessions
List all active sessions.

## Testing

Using curl:
```bash
# Send a message
curl -X POST http://localhost:5001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel tired and sad"}'

# Get session summary
curl http://localhost:5001/api/session/20250121_143022/summary
```

Using Python:
```python
import requests

response = requests.post('http://localhost:5001/api/chat', json={
    'message': 'I have been feeling down lately',
    'user_id': 'user123'
})

print(response.json())
# Output: {'text': '...', 'emotion': 'sad', 'session_id': '...', ...}
```

## Emotion Values

- `neutral`: Factual information, questions, general discussion
- `sad`: Empathy for difficult feelings, discussing challenges
- `happy`: Encouragement, celebrating progress, providing hope
