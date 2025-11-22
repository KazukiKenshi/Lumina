import React, { createContext, useContext, useRef } from 'react';

// Maintains a rolling window of recent emotion frames with probabilities
// Provides summary (dominant emotion + averaged probabilities)

const EmotionContext = createContext(null);

export const EmotionProvider = ({ children, windowMs = 30000 }) => {
  const framesRef = useRef([]); // { ts, emotion, probabilities }

  const addFrame = (emotion, probabilities) => {
    const ts = Date.now();
    framesRef.current.push({ ts, emotion, probabilities });
    // Purge old frames
    const cutoff = ts - windowMs;
    framesRef.current = framesRef.current.filter(f => f.ts >= cutoff);
  };

  const getSummary = () => {
    const now = Date.now();
    const cutoff = now - windowMs;
    const windowFrames = framesRef.current.filter(f => f.ts >= cutoff);
    if (windowFrames.length === 0) {
      return { dominantEmotion: 'neutral', averageProbabilities: null, frameCount: 0 };
    }
    // Aggregate probabilities if present
    let sum = { Happiness: 0, Neutral: 0, Sadness: 0 };
    let probCount = 0;
    const emotionCounts = { happy: 0, neutral: 0, sad: 0 };
    windowFrames.forEach(f => {
      // Count emotion tokens
      if (f.emotion) {
        if (f.emotion === 'happy') emotionCounts.happy++;
        else if (f.emotion === 'sad') emotionCounts.sad++;
        else emotionCounts.neutral++;
      }
      if (f.probabilities) {
        Object.keys(sum).forEach(k => {
          if (f.probabilities[k] !== undefined) {
            sum[k] += f.probabilities[k];
          }
        });
        probCount++;
      }
    });
    let dominantEmotion = 'neutral';
    if (emotionCounts.happy >= emotionCounts.neutral && emotionCounts.happy >= emotionCounts.sad) dominantEmotion = 'happy';
    else if (emotionCounts.sad >= emotionCounts.neutral && emotionCounts.sad >= emotionCounts.happy) dominantEmotion = 'sad';
    else dominantEmotion = 'neutral';

    let avg = null;
    if (probCount > 0) {
      avg = Object.fromEntries(Object.keys(sum).map(k => [k, sum[k] / probCount]));
    }
    return { dominantEmotion, averageProbabilities: avg, frameCount: windowFrames.length };
  };

  return (
    <EmotionContext.Provider value={{ addFrame, getSummary }}>
      {children}
    </EmotionContext.Provider>
  );
};

export const useEmotion = () => {
  const ctx = useContext(EmotionContext);
  if (!ctx) throw new Error('useEmotion must be used within EmotionProvider');
  return ctx;
};
