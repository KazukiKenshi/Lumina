import React, { useEffect, useRef, useState } from 'react';
import './UnityPlayer.css';

// Global reference to unity instance for external components
export let unityInstance = null;
let isUnityLoading = false;

const UnityPlayer = () => {
  const canvasRef = useRef(null);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState(null);
  const unityInstanceRef = useRef(null);
  const scriptRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || isUnityLoading) return;

    isUnityLoading = true;
    const buildUrl = `${process.env.PUBLIC_URL}/Build`;
    const loaderUrl = `${buildUrl}/build.loader.js`;
    
    const config = {
      arguments: [],
      dataUrl: `${buildUrl}/build.data`,
      frameworkUrl: `${buildUrl}/build.framework.js`,
      codeUrl: `${buildUrl}/build.wasm`,
      streamingAssetsUrl: 'StreamingAssets',
      companyName: 'DefaultCompany',
      productName: 'Lumina-frontend',
      productVersion: '0.1.0',
    };

    // Set canvas to full viewport size
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Handle window resize
    const handleResize = () => {
      if (canvas) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      }
    };
    window.addEventListener('resize', handleResize);

    // Load the Unity loader script
    const script = document.createElement('script');
    script.src = loaderUrl;
    script.async = true;
    scriptRef.current = script;
    
    script.onload = () => {
      if (window.createUnityInstance && canvas) {
        window.createUnityInstance(canvas, config, (progress) => {
          setLoadingProgress(Math.round(progress * 100));
        })
          .then((unityInstanceResult) => {
            if (unityInstanceResult) {
              unityInstanceRef.current = unityInstanceResult;
              unityInstance = unityInstanceResult;
              setIsLoaded(true);
              setLoadingProgress(100);
            }
          })
          .catch((message) => {
            setError(message);
            console.error('Unity loading error:', message);
            isUnityLoading = false;
          });
      }
    };

    script.onerror = () => {
      setError('Failed to load Unity loader script');
      isUnityLoading = false;
    };

    document.body.appendChild(script);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      
      // Clean up Unity instance first
      if (unityInstanceRef.current) {
        try {
          const instance = unityInstanceRef.current;
          unityInstanceRef.current = null;
          unityInstance = null;
          
          if (typeof instance.Quit === 'function') {
            Promise.resolve(instance.Quit()).catch(() => {
              // Ignore errors during quit
            });
          }
        } catch (err) {
          console.warn('Unity cleanup error (ignored):', err);
        }
      }
      
      // Remove script after Unity cleanup
      setTimeout(() => {
        if (scriptRef.current && document.body.contains(scriptRef.current)) {
          document.body.removeChild(scriptRef.current);
        }
        isUnityLoading = false;
      }, 100);
    };
  }, []);

  const handleFullscreen = () => {
    if (unityInstanceRef.current) {
      unityInstanceRef.current.SetFullscreen(1);
    }
  };

  const sendToUnity = (gameObjectName, methodName, value) => {
    if (unityInstanceRef.current) {
      unityInstanceRef.current.SendMessage(gameObjectName, methodName, value);
      console.log(`Sent to Unity: ${gameObjectName}.${methodName}(${value})`);
    } else {
      console.warn('Unity instance not loaded yet');
    }
  };

  const triggerExpression = (expression) => {
    if(expression == 'happy') sendToUnity('ReactBridge', 'TriggerHappy', "");
    else if(expression == 'neutral') sendToUnity('ReactBridge', 'TriggerNeutral', "");
    else if(expression == 'sad') sendToUnity('ReactBridge', 'TriggerSad', "");
  };

  const toggleTalking = (isTalking) => {
    sendToUnity('ReactBridge', 'SetBool', isTalking ? 'true' : 'false');
  };

  return (
    <div className="unity-container">
      <canvas 
        ref={canvasRef} 
        id="unity-canvas"
        tabIndex="-1"
      />
      
      {!isLoaded && !error && (
        <div className="unity-loading-bar">
          <div className="unity-logo"></div>
          <div className="unity-progress-bar-empty">
            <div 
              className="unity-progress-bar-full"
              style={{ width: `${loadingProgress}%` }}
            ></div>
          </div>
          <div className="loading-text">{loadingProgress}%</div>
        </div>
      )}

      {error && (
        <div className="unity-error">
          <p>Error loading Unity application:</p>
          <p>{error}</p>
        </div>
      )}

      {isLoaded && (
        <div className="unity-controls">
          <div className="control-group">
            <div className="control-label">Expression Triggers:</div>
            <div className="button-group">
              <button 
                className="control-button happy"
                onClick={() => triggerExpression('happy')}
                title="Trigger Happy Expression"
              >
                😊 Happy
              </button>
              <button 
                className="control-button neutral"
                onClick={() => triggerExpression('neutral')}
                title="Trigger Neutral Expression"
              >
                😐 Neutral
              </button>
              <button 
                className="control-button sad"
                onClick={() => triggerExpression('sad')}
                title="Trigger Sad Expression"
              >
                😢 Sad
              </button>
            </div>
          </div>
          
          <div className="control-group">
            <div className="control-label">Talking State:</div>
            <div className="button-group">
              <button 
                className="control-button talking-on"
                onClick={() => toggleTalking(true)}
                title="Start Talking"
              >
                🗣️ Start Talking
              </button>
              <button 
                className="control-button talking-off"
                onClick={() => toggleTalking(false)}
                title="Stop Talking"
              >
                🤐 Stop Talking
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default UnityPlayer;
