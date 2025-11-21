import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './VoiceRecorder.css';

const VoiceRecorder = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState('');
  
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check if browser supports Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcriptPiece = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcriptPiece + ' ';
          } else {
            interimTranscript += transcriptPiece;
          }
        }

        setTranscript(finalTranscript || interimTranscript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setStatus(`Error: ${event.error}`);
        setIsRecording(false);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    } else {
      setStatus('Speech Recognition not supported in this browser');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startRecording = () => {
    if (recognitionRef.current) {
      setTranscript('');
      setStatus('Listening...');
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const stopRecording = async () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsRecording(false);
      setStatus('Processing...');
      
      if (transcript.trim()) {
        await sendToAPI(transcript);
      } else {
        setStatus('No speech detected');
      }
    }
  };

  const sendToAPI = async (text) => {
    setIsProcessing(true);
    
    try {
      // Send to backend API endpoint
      const response = await axios.post('http://localhost:5000/api/process-speech', {
        transcript: text,
        timestamp: new Date().toISOString(),
        userId: 'user123', // Replace with actual user ID
        sessionId: sessionStorage.getItem('sessionId') || null
      });

      console.log('API Response:', response.data);
      
      // Store session ID if provided
      if (response.data.sessionId) {
        sessionStorage.setItem('sessionId', response.data.sessionId);
      }

      // Play the audio response if available
      if (response.data.audio) {
        playAudio(response.data.audio);
      }

      setStatus(`Response: ${response.data.response}`);
      
      // Clear transcript after showing response
      setTimeout(() => {
        setTranscript('');
        setStatus('');
      }, 5000);
      
    } catch (error) {
      console.error('API Error:', error);
      setStatus('Failed to process request');
    } finally {
      setIsProcessing(false);
    }
  };

  const playAudio = (audioBase64) => {
    try {
      const audio = new Audio(audioBase64);
      audio.play().catch(err => {
        console.error('Error playing audio:', err);
      });
    } catch (error) {
      console.error('Error creating audio:', error);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="voice-recorder">
      <button 
        className={`record-button ${isRecording ? 'recording' : ''}`}
        onClick={toggleRecording}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <span className="spinner"></span>
        ) : (
          <>
            <span className="mic-icon">🎤</span>
            <span className="button-label">
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </span>
          </>
        )}
      </button>

      {status && (
        <div className={`status-message ${status.includes('Error') ? 'error' : ''}`}>
          {status}
        </div>
      )}

      {transcript && (
        <div className="transcript-box">
          <div className="transcript-label">Transcript:</div>
          <div className="transcript-text">{transcript}</div>
        </div>
      )}
    </div>
  );
};

export default VoiceRecorder;
