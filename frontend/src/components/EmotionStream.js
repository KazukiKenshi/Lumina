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
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const { token } = useAuth();
  const { addFrame } = useEmotion();
  const API_URL = process.env.EMOTION_RECOGNITION_API || '';

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
      const resp = await axios.post(`${API_URL}/api/emotion-frame`, { image_base64 }, { headers });
      const em = resp.data.emotion || 'neutral';
      // Store frame in context window (no UI display)
      addFrame(em, resp.data.probabilities || null);
        // NOTE: Unity expression triggers disabled per requirement; only chatbot responses drive expressions.
    } catch (err) {
      console.warn('Emotion frame request failed:', err.message);
    }
  };

  // Do not render any visible UI; keep hidden video/canvas for capture
  return (
    <div style={{ display: 'none' }}>
      <video ref={videoRef} muted playsInline />
      <canvas ref={canvasRef} />
    </div>
  );
};

export default EmotionStream;
