const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const fetch = require('node-fetch');

// ---------------------------------------------
// Centralized backend console logging (tee)
// Captures all console output to a daily log file while preserving original console behavior.
// ---------------------------------------------
const originalConsole = { ...console };
const logDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}
function currentDateStamp() {
  const d = new Date();
  return d.toISOString().slice(0, 10); // YYYY-MM-DD
}
function timestamp() {
  return new Date().toISOString();
}
function getLogStream() {
  const filePath = path.join(logDir, `backend-terminal-${currentDateStamp()}.log`);
  return fs.createWriteStream(filePath, { flags: 'a' });
}
let stream = getLogStream();
// Rotate file if date changes during long runtimes (checked every log call)
let currentDate = currentDateStamp();
function ensureStream() {
  const today = currentDateStamp();
  if (today !== currentDate) {
    stream.end();
    stream = getLogStream();
    currentDate = today;
  }
}
function writeLine(level, args) {
  ensureStream();
  try {
    const line = `[${timestamp()}] [${level.toUpperCase()}] ` + args.map(a => {
      if (a instanceof Error) return a.stack || a.toString();
      if (typeof a === 'object') {
        try { return JSON.stringify(a); } catch { return String(a); }
      }
      return String(a);
    }).join(' ') + '\n';
    stream.write(line);
  } catch (e) {
    originalConsole.error('Logging write failed', e);
  }
}
['log', 'info', 'warn', 'error'].forEach(level => {
  console[level] = (...args) => {
    writeLine(level, args);
    originalConsole[level](...args);
  };
});
process.on('uncaughtException', err => {
  console.error('Uncaught Exception:', err);
});
process.on('unhandledRejection', reason => {
  console.error('Unhandled Rejection:', reason);
});

console.log('[BOOT] Backend logging initialized');
const axios = require('axios');
const { router: authRouter, authenticateToken } = require('./auth');
const connectDB = require('./config/database');
const ChatHistory = require('./models/ChatHistory');

require('dotenv').config();

// Connect to MongoDB
connectDB();

const app = express();
const PORT = process.env.SERVER_PORT || 5000;


//Middlewares

// Parse ALLOWED_ORIGINS from environment variable
const allowedOrigins = process.env.ALLOWED_ORIGINS 
  ? process.env.ALLOWED_ORIGINS.split(',').map(origin => origin.trim())
  : ['http://localhost:3000', 'http://localhost:8001', 'http://localhost:3001']; // fallback for development

// Configure CORS
app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (like mobile apps or curl)
    if (!origin) return callback(null, true);
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Auth routes
app.use('/api/auth', authRouter);

// Configuration - Update these URLs with your actual API endpoints
const CHATBOT_API_URL = process.env.CHATBOT_API_URL || 'https://jsonplaceholder.typicode.com/posts';
// const TTS_API_URL = process.env.TTS_API_URL || 'http://localhost:8000/tts'; // Your TTS service
const EMOTION_API_URL = process.env.EMOTION_API_URL || 'http://localhost:8001/predict';

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', message: 'Lumina Backend Server is running' });
});

// Main endpoint: Receive STT text, process through chatbot, convert to speech
app.post('/api/process-speech', authenticateToken, async (req, res) => {
  try {
    const { transcript, sessionId, emotionContext } = req.body;

    if (!transcript) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    console.log(`[${new Date().toISOString()}] Received transcript:`, transcript);
    if (emotionContext) {
      console.log('Received emotionContext:', emotionContext);
    }

    // Use user email as userId
    const userEmail = req.user.email;

    // Step 1: Send transcript to chatbot API
    console.log('Step 1: Sending to chatbot API...');
    let emotionTag = 'neutral';
    if (emotionContext && emotionContext.dominantEmotion) {
      emotionTag = emotionContext.dominantEmotion;
    }
    // Format: '[{emotion} face] transcript'
    const augmentedTranscript = `[${emotionTag} face] ${transcript}`;
    const chatbotResponse = await sendToChatbot(augmentedTranscript, userEmail, sessionId);
    console.log('Chatbot response:', chatbotResponse);

    // Step 2: Save chat history to database (store user dominant emotion if available)
    console.log('Step 2: Saving chat history...');
    // Normalize to JSON string with text + emotion for persistence and TTS
    const normalizedEmotion = (chatbotResponse && chatbotResponse.expression) ? String(chatbotResponse.expression).toLowerCase() : 'neutral';
    const assistantPayload = JSON.stringify({ text: chatbotResponse?.text || String(chatbotResponse || ''), emotion: normalizedEmotion });
    const userEmotionForDB = (emotionContext && emotionContext.dominantEmotion) ? emotionContext.dominantEmotion : 'neutral';
    await saveChatHistory(userEmail, transcript, assistantPayload, userEmotionForDB);

    // Step 3: Send chatbot response to TTS API
    console.log('Step 3: Converting to speech...');
    const audioBase64 = await convertToSpeech(assistantPayload);
    console.log('Audio generated successfully');

    // Step 4: Send response back to frontend
    res.json({
      success: true,
      transcript: transcript,
      response: chatbotResponse?.text || String(chatbotResponse || ''),
      emotion: normalizedEmotion,
      audio: audioBase64,
      sessionId: chatbotResponse?.sessionId || sessionId || null,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error processing speech:', error.message);
    res.status(500).json({
      error: 'Failed to process speech',
      message: error.message
    });
  }
});

// Function to send transcript to chatbot API
async function sendToChatbot(transcript, userId = 'default', sessionId = null) {
  try {
    // Replace this with your actual chatbot API call
    const response = await axios.post(CHATBOT_API_URL, {
      message: transcript,
      user_id: userId,
      session_id: sessionId || generateSessionId(),
      timestamp: new Date().toISOString()
    }, {
      timeout: 30000, // 30 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // Extract fields from chatbot API; fallbacks for different schemas
    const data = response.data || {};
    const text = data.text || data.response || data.message || data.content || `The server didn't respond.`;
    const expression = (data.expression || data.emotion || data.mood || 'neutral');
    const out = {
      text: typeof text === 'string' ? text : JSON.stringify(text),
      expression: typeof expression === 'string' ? expression : String(expression),
      sessionId: data.sessionId || data.session_id || null
    };
    return out;
  } catch (error) {
    console.error('Chatbot API error:', error.message);
    // Return a fallback structured response
    return {
      text: `I'm sorry, I'm having trouble processing your request right now. You said: "${transcript}"`,
      expression: 'neutral',
      sessionId: sessionId || null
    };
  }
}

// Function to convert text to speech and return base64
async function convertToSpeech(text) {
  try {
    // Extract emotion if text is JSON
    let promptText = text;
    let emotion = 'neutral';
    
    try {
      const parsedText = JSON.parse(text);
      if (parsedText.text && parsedText.emotion) {
        promptText = parsedText.text;
        emotion = parsedText.emotion;
      }
    } catch (e) {
      // Not JSON, use as is
    }

    // Dynamically import the ESM-only Gradio client (works from CommonJS)
    let ClientLib;
    try {
      const mod = await import('@gradio/client');
      // handle named export or default export
      ClientLib = mod.Client ?? mod.default ?? mod;
    } catch (e) {
      console.error('Failed to import @gradio/client', e);
      return null;
    }

    // Connect to Gradio TTS client
    const client = await ClientLib.connect("NihalGazi/Text-To-Speech-Unlimited");
    const result = await client.predict("/text_to_speech_app", {
      prompt: promptText,
      voice: "sage",
      emotion: emotion,
      use_random_seed: true,
      specific_seed: 3
    });

    console.log('Gradio TTS result:', result.data);

    // Check if we got audio data back
    if (result.data && result.data[0]) {
      const audioUrl = result.data[0].url;
      
      // Fetch the audio file and convert to base64
      const audioResponse = await axios.get(audioUrl, {
        responseType: 'arraybuffer'
      });
      
      const audioBuffer = Buffer.from(audioResponse.data);
      const audioBase64 = audioBuffer.toString('base64');
      
      return `data:audio/wav;base64,${audioBase64}`;
    }
    
    return null;
  } catch (error) {
    console.error('TTS API error:', error.message);
    
    return null;
  }
}

// Helper function to generate session ID
function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Separate endpoint for TTS only (if needed)
app.post('/api/text-to-speech', async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const audioBase64 = await convertToSpeech(text);

    if (!audioBase64) {
      return res.status(500).json({ error: 'Failed to generate audio' });
    }

    res.json({
      success: true,
      audio: audioBase64,
      text: text
    });
  } catch (error) {
    console.error('TTS error:', error.message);
    res.status(500).json({
      error: 'Failed to convert text to speech',
      message: error.message
    });
  }
});

// Separate endpoint for chatbot only (if needed)
app.post('/api/chatbot', async (req, res) => {
  try {
    const { message, userId, sessionId } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    const response = await sendToChatbot(message, userId, sessionId);

    res.json({
      success: true,
      message: message,
      response: response,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Chatbot error:', error.message);
    res.status(500).json({
      error: 'Failed to get chatbot response',
      message: error.message
    });
  }
});

// Webcam emotion frame endpoint (expects base64 image string)
app.post('/api/emotion-frame', authenticateToken, async (req, res) => {
  try {
    const { image_base64 } = req.body;
    if (!image_base64) {
      return res.status(400).json({ error: 'image_base64 is required' });
    }
    // Forward to Python emotion API
    let pyResp;
    let data = {};
    let apiError = null;
    try {
      pyResp = await axios.post(EMOTION_API_URL, { image_base64 }, { timeout: 5000 });
      data = pyResp.data || {};
    } catch (err) {
      apiError = err;
      // Do not print error, fallback to neutral
    }
    let rawEmotion = (data.emotion || '').toLowerCase();
    let normalizedEmotion = 'neutral';
    if (rawEmotion.includes('hap')) normalizedEmotion = 'happy';
    else if (rawEmotion.includes('sad')) normalizedEmotion = 'sad';
    // If API failed or emotion not returned, use neutral
    if (apiError || !rawEmotion) {
      normalizedEmotion = 'neutral';
      data.probabilities = null;
    }
    // Build response
    res.json({
      success: true,
      emotion: normalizedEmotion,
      probabilities: data.probabilities || null,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    // Fallback to neutral if any error
    res.json({
      success: true,
      emotion: 'neutral',
      probabilities: null,
      timestamp: new Date().toISOString()
    });
  }
});

// Function to save chat history to database
async function saveChatHistory(userId, userMessage, assistantResponse, userEmotion = 'neutral') {
  try {
    // Find or create chat history for today
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    let chatHistory = await ChatHistory.findOne({
      userId: userId,
      createdAt: { $gte: today }
    });

    if (!chatHistory) {
      // Create new chat history for today
      chatHistory = new ChatHistory({
        userId: userId,
        messages: []
      });
    }

    // Add user message
    chatHistory.messages.push({
      role: 'user',
      content: userMessage,
      emotion: userEmotion || 'neutral'
    });

    // Add assistant response (extract emotion if it's JSON)
    let assistantEmotion = 'neutral';
    let assistantContent = assistantResponse;

    try {
      const parsedResponse = JSON.parse(assistantResponse);
      if (parsedResponse.text && parsedResponse.emotion) {
        assistantContent = parsedResponse.text;
        assistantEmotion = parsedResponse.emotion;
      }
    } catch (e) {
      // Not JSON, use as is
    }

    chatHistory.messages.push({
      role: 'assistant',
      content: assistantContent,
      emotion: assistantEmotion
    });

    await chatHistory.save();
    console.log('Chat history saved successfully');
  } catch (error) {
    console.error('Error saving chat history:', error.message);
    // Don't throw - let the conversation continue even if saving fails
  }
}

// Get chat history for authenticated user
app.get('/api/chat-history', authenticateToken, async (req, res) => {
  try {
    const { limit = 10, skip = 0 } = req.query;

    // Use user email as userId
    const userEmail = req.user.email;

    const chatHistories = await ChatHistory.find({ userId: userEmail })
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip));

    res.json({
      success: true,
      chatHistories: chatHistories,
      count: chatHistories.length
    });
  } catch (error) {
    console.error('Error fetching chat history:', error.message);
    res.status(500).json({
      error: 'Failed to fetch chat history',
      message: error.message
    });
  }
});

// Get specific chat session by ID
app.get('/api/chat-history/:id', authenticateToken, async (req, res) => {
  try {
    // Use user email as userId
    const userEmail = req.user.email;
    const chatHistory = await ChatHistory.findOne({
      _id: req.params.id,
      userId: userEmail
    });

    if (!chatHistory) {
      return res.status(404).json({ error: 'Chat history not found' });
    }

    res.json({
      success: true,
      chatHistory: chatHistory
    });
  } catch (error) {
    console.error('Error fetching chat history:', error.message);
    res.status(500).json({
      error: 'Failed to fetch chat history',
      message: error.message
    });
  }
});

// Delete chat history by ID
app.delete('/api/chat-history/:id', authenticateToken, async (req, res) => {
  try {
    // Use user email as userId
    const userEmail = req.user.email;
    const result = await ChatHistory.findOneAndDelete({
      _id: req.params.id,
      userId: userEmail
    });

    if (!result) {
      return res.status(404).json({ error: 'Chat history not found' });
    }

    res.json({
      success: true,
      message: 'Chat history deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting chat history:', error.message);
    res.status(500).json({
      error: 'Failed to delete chat history',
      message: error.message
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`\n🚀 Lumina Backend Server is running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/health`);
  console.log(`🎤 Speech API: http://localhost:${PORT}/api/process-speech`);
  console.log(`💬 Chatbot API: ${CHATBOT_API_URL}`);
  console.log(`🔊 TTS API: http://localhost:${PORT}/api/text-to-speech\n`);
});

server.keepAliveTimeout = 120000;
server.headersTimeout = 120000;

module.exports = app;
