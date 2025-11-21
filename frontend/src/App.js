import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Home from './components/Home';
import Login from './components/Login';
import Register from './components/Register';
import PrivateRoute from './components/PrivateRoute';
import UnityPlayer from './components/UnityPlayer';
import ChatInput from './components/ChatInput';

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
                <UnityPage />
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
      <Link to="/" className="floating-back-button">← Home</Link>
      <button onClick={handleSignOut} className="floating-signout-button">
        Sign Out
      </button>
      <ChatInput />
      <UnityPlayer />
    </div>
  );
}

export default App;
