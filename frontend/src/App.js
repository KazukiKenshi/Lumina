import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Home from './components/Home';
import ChatHistoryPage from './components/ChatHistoryPage';
import Login from './components/Login';
import Register from './components/Register';
import PrivateRoute from './components/PrivateRoute';
import UnityPlayer from './components/UnityPlayer';
import ChatInput from './components/ChatInput';
import ChatOverlay from './components/ChatOverlay';
import EmotionStream from './components/EmotionStream';
import { ChatProvider } from './contexts/ChatContext';
import { EmotionProvider } from './contexts/EmotionContext';

function App() {
  console.log('📱 App component is rendering');
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/play" element={
              <PrivateRoute>
                <EmotionProvider>
                  <ChatProvider>
                    <UnityPage />
                  </ChatProvider>
                </EmotionProvider>
              </PrivateRoute>
            } />
            <Route path="/history" element={
              <PrivateRoute>
                <ChatHistoryPage />
              </PrivateRoute>
            } />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
}

function UnityPage() {
  console.log('🎮 UnityPage is rendering');
  const navigate = useNavigate();
  const { logout } = useAuth();

  const handleSignOut = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="unity-page">
      <Link to="/" className="floating-back-button">Home</Link>
      <Link to="/history" className="floating-back-button" style={{ left: '120px' }}>📜 History</Link>
      <button onClick={handleSignOut} className="floating-signout-button">
        Sign Out
      </button>
      <ChatOverlay />
      <EmotionStream />
      <ChatInput />
      <UnityPlayer />
    </div>
  );
}

export default App;
