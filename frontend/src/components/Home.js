import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Home.css';

function Home() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleGetStarted = () => {
    if (user) {
      navigate('/play');
    } else {
      navigate('/login');
    }
  };

  const handleSignOut = () => {
    logout();
  };

  return (
    <div className="home-container">
      <nav className="navbar">
        <div className="logo">Lumina</div>
        <div className="nav-links">
          {user ? (
            <>
              <Link to="/play" className="nav-link">Get Started</Link>
              <button onClick={handleSignOut} className="nav-link signout-btn">Sign Out</button>
            </>
          ) : (
            <Link to="/login" className="nav-link">Sign In</Link>
          )}
        </div>
      </nav>

      <section className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            Your Journey to <span className="highlight">Mental Wellness</span>
          </h1>
          <p className="hero-subtitle">
            Experience empathetic AI-driven support in an immersive, interactive environment designed for your mental wellbeing.
          </p>
          <div className="cta-buttons">
            <button onClick={handleGetStarted} className="primary-button">
              Begin Your Journey
            </button>
            <a href="#features" className="secondary-button">
              Learn More
            </a>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-shape shape-1"></div>
          <div className="floating-shape shape-2"></div>
          <div className="floating-shape shape-3"></div>
        </div>
      </section>

      <section id="features" className="features-section">
        <h2 className="section-title">Why Choose Lumina?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="feature-title">24/7 Support</h3>
            <p className="feature-description">
              Access compassionate AI-powered support anytime, anywhere. Your mental health companion is always ready to listen.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="feature-title">Emotion Recognition</h3>
            <p className="feature-description">
              Advanced AI detects and responds to your emotional state, providing personalized support tailored to your needs.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h3 className="feature-title">Private & Secure</h3>
            <p className="feature-description">
              Your conversations are confidential and secure. We prioritize your privacy with end-to-end protection.
            </p>
          </div>

          <div className="feature-card">
            <div className="feature-icon-wrapper">
              <svg className="feature-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
            </div>
            <h3 className="feature-title">Interactive Experience</h3>
            <p className="feature-description">
              Engage with a beautiful 3D environment that makes mental wellness support more immersive and engaging.
            </p>
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <p>© 2025 Lumina. Your Mental Wellness Companion.</p>
      </footer>
    </div>
  );
}

export default Home;
