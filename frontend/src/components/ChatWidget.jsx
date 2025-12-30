import React, { useState, useEffect, useRef } from 'react';
import { useChat } from '../services/chat-context';
import useChatHook from '../hooks/useChat';

const ChatWidget = ({ pageContext = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [selectedText, setSelectedText] = useState('');

  // Use the chat context and hook
  const {
    messages,
    sessionId,
    isLoading,
    error,
    updateContext
  } = useChat();

  const {
    sendMessage,
    startNewSession
  } = useChatHook();

  // Update page context when component mounts or when pageContext changes
  useEffect(() => {
    updateContext(pageContext);
  }, [pageContext, updateContext]);

  // Add event listener for text selection
  useEffect(() => {
    const handleSelection = () => {
      const selected = window.getSelection().toString().trim();
      setSelectedText(selected);
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputValue.trim() && sessionId) {
      try {
        // Send message using the hook, including selected text
        await sendMessage(inputValue, selectedText);
        setInputValue('');
        // Clear selected text after sending
        setSelectedText('');
      } catch (err) {
        console.error('Error sending message:', err);
        // Error is handled by the hook
      }
    }
  };

  return (
    <div className="chat-widget">
      <div className="chat-container">
        <div className="chat-header">
          <h3>Physical AI Book Assistant</h3>
          <button onClick={toggleChat} className="close-btn">X</button>
        </div>
        <div className="chat-messages">
          {error && (
            <div className="error-message">
              <span>{error}</span>
            </div>
          )}
          {selectedText && (
            <div className="selected-text-context">
              <strong>Selected text:</strong> "{selectedText}"
            </div>
          )}
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <span>{message.content}</span>
              {message.sources && message.sources.length > 0 && (
                <div className="sources">
                  <small>Sources: {message.sources.map(s => s.title).join(', ')}</small>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <span>Thinking...</span>
            </div>
          )}
        </div>
        <form onSubmit={handleSendMessage} className="chat-input-form">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about the book..."
            disabled={isLoading || !sessionId}
          />
          <button type="submit" disabled={isLoading || !sessionId}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatWidget;