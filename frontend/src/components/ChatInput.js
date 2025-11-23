import React, { useState, useRef, useEffect } from 'react';
// Add import for user context
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { unityInstance } from './UnityPlayer';
import { useChat } from '../contexts/ChatContext';
import './ChatInput.css';
import { useEmotion } from '../contexts/EmotionContext';

const ChatInput = () => {
    const { user } = useAuth(); // Assumes user object has an 'email' property
    const [inputText, setInputText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const recognitionRef = useRef(null);
    const inputRef = useRef(null);
    const { addMessage } = useChat();
    const { getSummary } = useEmotion();
    const API_URL = process.env.REACT_APP_BACKEND_URL || '';

    // Minimal SpeechRecognition setup
    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn('Speech recognition not supported');
            return;
        }
        const rec = new SpeechRecognition();
        rec.continuous = false;
        rec.interimResults = false;
        rec.lang = 'en-US';

        rec.onresult = (event) => {
            try {
                const transcript = event.results[0][0].transcript.trim();
                if (transcript) {
                    // Stop recording immediately so we don't get duplicate events
                    try { rec.stop(); } catch {}
                    // Submit directly with captured transcript (avoid waiting for state flush)
                    if (!isProcessing) {
                        // add user message then submit
                        addMessage('user', transcript);
                        handleSubmit({ preventDefault: () => {} }, transcript);
                    }
                    // Clear input for auto submission
                    setInputText('');
                }
            } catch (err) {
                console.error('SR parse error:', err);
            }
        };
        rec.onerror = (e) => console.error('SR error:', e.error);
        rec.onend = () => setIsRecording(false);
        recognitionRef.current = rec;
        return () => { try { rec.stop(); } catch {} };
    }, [isProcessing]);

    // Pre-flight microphone permission request
    const ensureMicPermission = async () => {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            return true;
        } catch (err) {
            console.error('Mic permission error:', err);
            return false;
        }
    };

    const handleSubmit = async (e, directText) => {
        e.preventDefault();
        const textToSend = (directText !== undefined ? directText : inputText).trim();
        if (!textToSend || isProcessing) return;

        setIsProcessing(true);
        // Clear input immediately for both manual and auto submissions
        setInputText('');

        // If this is a manual submit (not the STT path which already adds), push user message
        if (directText === undefined) {
            addMessage('user', textToSend);
        }

        try {
            // Reintroduce emotion context summary for LLM prompt engineering
            const summary = getSummary();
            const emotionContext = summary.frameCount > 0 ? {
                dominantEmotion: summary.dominantEmotion,
                averageProbabilities: summary.averageProbabilities,
                frameCount: summary.frameCount,
                windowMs: 30000
            } : null;
            const response = await axios.post(`${API_URL}/api/process-speech`, {
                transcript: textToSend,
                timestamp: new Date().toISOString(),
                userId: user?.email || 'anonymous',
                sessionId: sessionStorage.getItem('sessionId') || null,
                emotionContext
            });

            console.log('Response:', response.data);

            if (response.data.sessionId) {
                sessionStorage.setItem('sessionId', response.data.sessionId);
            }

            if (response.data.emotion && unityInstance) {
                try {
                    const em = String(response.data.emotion).toLowerCase();
                    let method = 'TriggerNeutral';
                    if (em.includes('happy') || em.includes('joy')) method = 'TriggerHappy';
                    else if (em.includes('sad') || em.includes('depress')) method = 'TriggerSad';
                    else if (em.includes('anger') || em.includes('angry')) method = 'TriggerSad'; // fallback mapping
                    else if (em.includes('fear') || em.includes('anxiety')) method = 'TriggerSad';
                    unityInstance.SendMessage('ReactBridge', method, "");
                } catch (err) {
                    console.warn('Emotion trigger failed:', err);
                }
            }

            if (response.data.audio) {
                console.log('Audio data received:', response.data.audio.substring(0, 100));

                // Send audio to Unity for playback
                if (unityInstance) {
                    try {
                        // Get only base64 part
                        const base64Audio = response.data.audio.includes(',')
                            ? response.data.audio.split(',')[1]
                            : response.data.audio;

                        // Convert to blob
                        const binary = atob(base64Audio);
                        const len = binary.length;
                        const bytes = new Uint8Array(len);
                        for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);

                        const blob = new Blob([bytes], { type: "audio/wav" });
                        const url = URL.createObjectURL(blob);  // <-- REAL URL (no invalid URI)

                        console.log("Blob URL:", url);

                        unityInstance.SendMessage("ReactBridge", "PlayAudioURL", url);
                    } catch (err) {
                        console.error("Error sending audio to Unity:", err);
                    }
                } else {
                    console.warn('Unity instance not available');
                }
            }

            // Add assistant message (response text may be direct string or object)
            const assistantText = typeof response.data.response === 'string' ? response.data.response : JSON.stringify(response.data.response);
            addMessage('assistant', assistantText);
            // Already cleared at start; no conditional reset needed
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    const toggleRecording = async () => {
        if (isRecording) {
            try { recognitionRef.current?.stop(); } catch {}
            setIsRecording(false);
            return;
        }
        const ok = await ensureMicPermission();
        if (!ok) return;
        if (!recognitionRef.current) return;
        setIsRecording(true);
        try { recognitionRef.current.start(); } catch (err) { console.error('SR start failed', err); }
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
                        className={`send-button ${isProcessing ? 'processing' : ''}`}
                        disabled={!inputText.trim() || isProcessing}
                        title="Send Message"
                    >
                        {isProcessing ? <span className="send-spinner" aria-label="Loading"></span> : '➤'}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default ChatInput;
