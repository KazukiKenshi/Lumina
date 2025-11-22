import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { useEmotion } from '../contexts/EmotionContext';
import { unityInstance } from './UnityPlayer';
import { useAuth } from '../contexts/AuthContext';

// Captures webcam frames at interval and sends to backend for emotion inference
const EmotionStream = ({ intervalMs = 2000, enabled = true }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [permission, setPermission] = useState(null); // null=unknown, true/false
  const [emotion, setEmotion] = useState('neutral');
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const { token } = useAuth();
  const { addFrame } = useEmotion();

  useEffect(() => {
    if (!enabled) return;
    let stream;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setPermission(true);
          setRunning(true);
        }
      } catch (err) {
        console.warn('Webcam permission denied:', err);
        setPermission(false);
        setError('Webcam not allowed');
      }
    })();
    return () => {
      setRunning(false);
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [enabled]);

  useEffect(() => {
    if (!running || !permission) return;
    const id = setInterval(() => {
      try {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const w = video.videoWidth || 320;
        const h = video.videoHeight || 240;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, w, h);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
        const base64 = dataUrl.split(',')[1];
        sendFrame(base64);
      } catch (err) {
        console.error('Frame capture failed:', err);
      }
    }, intervalMs);
    return () => clearInterval(id);
  }, [running, permission, intervalMs]);

  const sendFrame = async (image_base64) => {
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const resp = await axios.post('http://localhost:5000/api/emotion-frame', { image_base64 }, { headers });
      const em = resp.data.emotion || 'neutral';
      setEmotion(em);
      // Store frame in context window
      addFrame(em, resp.data.probabilities || null);
        // NOTE: Unity expression triggers disabled per requirement; only chatbot responses drive expressions.
    } catch (err) {
      console.warn('Emotion frame request failed:', err.message);
    }
  };

  return (
    <div style={{ position: 'fixed', bottom: '110px', right: '12px', zIndex: 50, fontSize: '12px', fontFamily: 'sans-serif', background: 'rgba(0,0,0,0.5)', color: '#fff', padding: '6px 8px', borderRadius: '6px' }}>
      <div style={{ marginBottom: '4px' }}>
        {permission === null && 'Requesting webcam...'}
        {permission === false && 'Webcam blocked'}
        {permission === true && `Webcam active (${intervalMs/1000}s)`}
      </div>
      <div>Live emotion: <strong style={{ color: emotion === 'happy' ? '#ffd76b' : emotion === 'sad' ? '#7fb0ff' : '#ccc' }}>{emotion}</strong></div>
      {error && <div style={{ color: '#ff8080' }}>{error}</div>}
      <video ref={videoRef} style={{ display: 'none' }} muted playsInline />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default EmotionStream;
