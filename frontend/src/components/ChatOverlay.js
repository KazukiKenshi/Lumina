import React from 'react';
import { useChat } from '../contexts/ChatContext';
import './ChatOverlay.css';

const ChatOverlay = () => {
  const { messages, overlayRef } = useChat();
  return (
    <div className="chat-overlay" ref={overlayRef}>
      {messages.length === 0 && (
        <div className="chat-overlay-empty">No messages yet. Speak or type to begin.</div>
      )}
      {messages.map(m => (
        <div key={m.id} className={`chat-line ${m.role}`}> 
          <span className="role-label">{m.role === 'user' ? 'You' : 'Lumina'}</span>
          <span className="content">{m.content}</span>
        </div>
      ))}
    </div>
  );
};

export default ChatOverlay;
