import React from 'react';
import { useChat } from '../contexts/ChatContext';
import './ChatOverlay.css';

const formatTime = (ts) => {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
};

const ChatOverlay = () => {
  const { messages, overlayRef } = useChat();
  return (
    <div className="chat-overlay" ref={overlayRef}>
      {messages.length === 0 && (
        <div className="chat-overlay-empty">No messages yet. Speak or type to begin.</div>
      )}
      {messages.map(m => {
        const isUser = m.role === 'user';
        return (
          <div key={m.id} className={`message-row ${isUser ? 'user' : 'assistant'}`}> 
            {isUser ? (
              <>
                <div className="bubble user">{m.content}</div>
                <span className="timestamp user-ts">{formatTime(m.timestamp)}</span>
              </>
            ) : (
              <>
                <div className="bubble assistant">{m.content}</div>
                <span className="timestamp assistant-ts">{formatTime(m.timestamp)}</span>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default ChatOverlay;
