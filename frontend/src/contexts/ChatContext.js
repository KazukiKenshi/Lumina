import React, { createContext, useContext, useState, useRef, useEffect } from 'react';

const ChatContext = createContext(null);

export function ChatProvider({ children }) {
  const [messages, setMessages] = useState([]); // {id, role:'user'|'assistant', content, timestamp}
  const overlayRef = useRef(null);

  const addMessage = (role, content) => {
    setMessages(prev => [...prev, { id: prev.length + 1, role, content, timestamp: Date.now() }]);
  };

  // Auto scroll overlay when messages change
  useEffect(() => {
    if (overlayRef.current) {
      overlayRef.current.scrollTop = overlayRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <ChatContext.Provider value={{ messages, addMessage, overlayRef }}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChat must be used within ChatProvider');
  return ctx;
}
