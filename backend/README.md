# Lumina Backend Server

Express.js backend server for the Lumina mental health application. Handles Speech-to-Text processing, chatbot API communication, and Text-to-Speech conversion.

## Features

- 🎤 **STT Processing**: Receives transcribed text from frontend
- 🤖 **Chatbot Integration**: Forwards messages to chatbot API
- 🔊 **TTS Conversion**: Converts text responses to audio (base64)
- 🔄 **Complete Pipeline**: STT → Chatbot → TTS → Frontend

## Installation

```bash
cd backend
npm install
```

## Configuration

Create a `.env` file with the following variables:

```env
PORT=5000
CHATBOT_API_URL=http://your-chatbot-api-url
TTS_API_URL=http://your-tts-api-url
```

## Running the Server

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

The server will run on `http://localhost:5000`

## API Endpoints

### 1. Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "message": "Lumina Backend Server is running"
}
```

### 2. Process Speech (Main Endpoint)
```
POST /api/process-speech
```

**Request Body:**
```json
{
  "transcript": "Hello, how are you?",
  "userId": "user123",
  "sessionId": "session456"
}
```

**Response:**
```json
{
  "success": true,
  "transcript": "Hello, how are you?",
  "response": "I'm doing well, thank you for asking!",
  "audio": "data:audio/mp3;base64,//uQx...",
  "timestamp": "2025-11-21T12:00:00.000Z"
}
```

### 3. Chatbot Only
```
POST /api/chatbot
```

**Request Body:**
```json
{
  "message": "Tell me about anxiety",
  "userId": "user123",
  "sessionId": "session456"
}
```

### 4. Text-to-Speech Only
```
POST /api/text-to-speech
```

**Request Body:**
```json
{
  "text": "This is a test message"
}
```

## Integration with Frontend

Update your `VoiceRecorder.js` to send to the backend:

```javascript
const response = await axios.post('http://localhost:5000/api/process-speech', {
  transcript: text,
  timestamp: new Date().toISOString(),
  userId: 'user123'
});

// Play the audio
const audio = new Audio(response.data.audio);
audio.play();
```

## Architecture

```
Frontend (STT) 
    ↓
Backend (/api/process-speech)
    ↓
Chatbot API (Get text response)
    ↓
TTS API (Convert to audio)
    ↓
Backend (Convert to base64)
    ↓
Frontend (Play audio)
```

## Error Handling

The server includes comprehensive error handling:
- Invalid requests (400)
- API failures (500)
- Timeout handling
- Fallback responses

## Notes

- Audio is returned as base64-encoded data URI
- Session IDs are auto-generated if not provided
- CORS is enabled for frontend integration
- Timeouts: 30s for chatbot, 60s for TTS

## License

Part of the Lumina project.
