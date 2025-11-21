import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './ChatInput.css';

const ChatInput = () => {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const recognitionRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    // Initialize Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('Speech result:', transcript);
        setInputText(transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
      };

      recognitionRef.current.onend = () => {
        console.log('Speech recognition ended');
        setIsRecording(false);
      };
    } else {
      console.warn('Speech recognition not supported');
    }

    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          // Ignore if already stopped
        }
      }
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!inputText.trim() || isProcessing) return;

    setIsProcessing(true);

    try {
      const response = await axios.post('http://localhost:5000/api/process-speech', {
        transcript: inputText.trim(),
        timestamp: new Date().toISOString(),
        userId: 'user123',
        sessionId: sessionStorage.getItem('sessionId') || null
      });

      console.log('Response:', response.data);

      if (response.data.sessionId) {
        sessionStorage.setItem('sessionId', response.data.sessionId);
      }

      if (response.data.audio) {
        const audio = new Audio(response.data.audio);
        audio.play().catch(err => console.error('Error playing audio:', err));
      }

      // Clear input after successful send
      setInputText('');
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleRecording = () => {
    console.log('Toggle recording clicked');
    if (isRecording) {
      console.log('Stopping recording');
      recognitionRef.current?.stop();
      setIsRecording(false);
    } else {
      if (recognitionRef.current) {
        console.log('Starting recording');
        recognitionRef.current.start();
        setIsRecording(true);
      } else {
        console.error('Speech recognition not available');
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-input-container">
      <form onSubmit={handleSubmit} className="chat-input-form">
        <div className="input-wrapper">
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message or use the microphone..."
            className="chat-input-field"
            disabled={false}
            autoComplete="off"
            spellCheck="false"
          />

          <button
            type="button"
            onClick={toggleRecording}
            className={`mic-button ${isRecording ? 'recording' : ''}`}
            disabled={isProcessing}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            {isRecording ? '🎤' : '🎙️'}
          </button>

          <button
            type="submit"
            className="send-button"
            disabled={!inputText.trim() || isProcessing}
            title="Send Message"
          >
            {isProcessing ? '⏳' : '➤'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput;
