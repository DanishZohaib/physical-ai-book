import React, { useState, useEffect } from 'react';
import ChatWidget from './ChatWidget';

const ChatModal = ({ pageContext = '' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  // Handle keyboard shortcut (Ctrl/Cmd + Shift + C)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        setIsVisible(!isVisible);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isVisible]);

  // Close modal when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      const chatContainer = document.querySelector('.chat-modal-container');
      if (chatContainer && !chatContainer.contains(e.target) && isVisible && !isMinimized) {
        setIsVisible(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isVisible, isMinimized]);

  // Close when navigating to a new page (in a real app, this would be handled by the router)
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Store current state in localStorage for persistence
      localStorage.setItem('chatModalState', JSON.stringify({
        isVisible: isVisible,
        isMinimized: isMinimized
      }));
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isVisible, isMinimized]);

  // Restore state from localStorage when component mounts
  useEffect(() => {
    const savedState = localStorage.getItem('chatModalState');
    if (savedState) {
      const { isVisible: savedVisible, isMinimized: savedMinimized } = JSON.parse(savedState);
      setIsVisible(savedVisible);
      setIsMinimized(savedMinimized);
    }
  }, []);

  const toggleVisibility = () => {
    setIsVisible(!isVisible);
    if (!isVisible) {
      setIsMinimized(false);
    }
  };

  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  return (
    <>
      {/* Floating button to open chat */}
      {!isVisible && (
        <button
          className="chat-float-button"
          onClick={toggleVisibility}
          title="Open AI Assistant (Ctrl+Shift+C)"
        >
          💬
        </button>
      )}

      {/* Chat modal */}
      {isVisible && (
        <div className={`chat-modal-container ${isMinimized ? 'minimized' : ''}`}>
          <div className="chat-modal-header">
            <button
              className="minimize-btn"
              onClick={toggleMinimize}
              title={isMinimized ? "Expand" : "Minimize"}
            >
              {isMinimized ? "+" : "−"}
            </button>
            <h3>Physical AI Book Assistant</h3>
            <button
              className="close-btn"
              onClick={toggleVisibility}
              title="Close (Esc)"
            >
              ×
            </button>
          </div>

          {!isMinimized && (
            <div className="chat-modal-content">
              <ChatWidget pageContext={pageContext} />
            </div>
          )}
        </div>
      )}

      {/* Global styles for the chat modal */}
      <style jsx>{`
        .chat-float-button {
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 60px;
          height: 60px;
          border-radius: 50%;
          background: #4f46e5;
          color: white;
          border: none;
          font-size: 24px;
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          z-index: 10000;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s ease;
        }

        .chat-float-button:hover {
          transform: scale(1.1);
          box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }

        .chat-modal-container {
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 400px;
          height: 500px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
          z-index: 10000;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .chat-modal-container.minimized {
          width: 300px;
          height: 40px;
        }

        .chat-modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 15px;
          background: #4f46e5;
          color: white;
          cursor: move;
        }

        .chat-modal-header h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }

        .chat-modal-header button {
          background: none;
          border: none;
          color: white;
          font-size: 18px;
          cursor: pointer;
          padding: 0 5px;
          margin: 0 2px;
          border-radius: 3px;
        }

        .chat-modal-header button:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .chat-modal-content {
          flex: 1;
          overflow: hidden;
        }

        /* Override ChatWidget styles to fit in modal */
        .chat-modal-content .chat-container {
          height: 100%;
          border: none;
          box-shadow: none;
        }

        .chat-modal-content .chat-messages {
          height: calc(100% - 100px);
          overflow-y: auto;
        }

        .chat-modal-content .chat-input-form {
          position: absolute;
          bottom: 0;
          width: calc(100% - 30px);
          padding: 10px;
        }
      `}</style>
    </>
  );
};

export default ChatModal;