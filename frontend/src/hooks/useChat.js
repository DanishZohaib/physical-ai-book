import { useState, useEffect, useCallback } from 'react';
import ChatAPI from '../services/api';
import { useChat as useChatContext } from '../services/chat-context';

/**
 * Custom hook for managing chat functionality
 * Provides methods for sending messages, managing sessions, and handling chat state
 */
const useChat = () => {
  const {
    sessionId,
    messages,
    isLoading,
    error,
    pageContext,
    setSessionId,
    addMessage,
    setLoading,
    setError,
    setPageContext,
    updatePageContext
  } = useChatContext();

  // Initialize session when the hook is used
  useEffect(() => {
    const initSession = async () => {
      if (!sessionId) {
        try {
          const sessionData = await ChatAPI.createSession({
            page: pageContext,
            user_agent: 'docusaurus-client'
          });
          setSessionId(sessionData.session_id);
        } catch (err) {
          console.error('Failed to create session:', err);
          setError('Failed to initialize chat session. Some features may not work properly.');
        }
      }
    };

    initSession();
  }, [sessionId, pageContext, setSessionId, setError]);

  // Function to send a message
  const sendMessage = useCallback(async (question, selectedText = null) => {
    if (!question.trim() || !sessionId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Add user message to chat
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: question,
        timestamp: new Date().toISOString(),
        selectedText: selectedText || null
      };
      addMessage(userMessage);

      // Send message to backend API
      const response = await ChatAPI.sendMessage(
        question,
        sessionId,
        pageContext,
        selectedText
      );

      // Add assistant response to chat
      const assistantMessage = {
        id: response.id,
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date().toISOString()
      };
      addMessage(assistantMessage);

      return response;
    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to get response. Please try again.');

      // Add error message to chat
      const errorMessage = {
        id: Date.now(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
        timestamp: new Date().toISOString()
      };
      addMessage(errorMessage);

      throw err;
    } finally {
      setLoading(false);
    }
  }, [sessionId, pageContext, addMessage, setLoading, setError]);

  // Function to update the page context when user navigates
  const updateContext = useCallback((newPageContext) => {
    updatePageContext(newPageContext);
  }, [updatePageContext]);

  // Function to clear the chat
  const clearChat = useCallback(() => {
    // In a real implementation, you might want to clear messages in the context
    // For now, this is a placeholder
  }, []);

  // Function to start a new session
  const startNewSession = useCallback(async (initialContext = null) => {
    try {
      const contextToUse = initialContext || {
        page: pageContext,
        user_agent: 'docusaurus-client'
      };

      const sessionData = await ChatAPI.createSession(contextToUse);
      setSessionId(sessionData.session_id);
      return sessionData;
    } catch (err) {
      console.error('Failed to create new session:', err);
      setError('Failed to create new session.');
      throw err;
    }
  }, [pageContext, setSessionId, setError]);

  return {
    // State
    sessionId,
    messages,
    isLoading,
    error,
    pageContext,

    // Actions
    sendMessage,
    updateContext,
    clearChat,
    startNewSession,
  };
};

export default useChat;