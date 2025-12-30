import React, { createContext, useContext, useReducer, useEffect } from 'react';

// Define the initial state
const initialState = {
  sessionId: null,
  messages: [],
  isLoading: false,
  error: null,
  pageContext: '',
  isVisible: false,
  isMinimized: false,
};

// Define action types
const actionTypes = {
  SET_SESSION_ID: 'SET_SESSION_ID',
  ADD_MESSAGE: 'ADD_MESSAGE',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_PAGE_CONTEXT: 'SET_PAGE_CONTEXT',
  SET_VISIBILITY: 'SET_VISIBILITY',
  SET_MINIMIZED: 'SET_MINIMIZED',
  CLEAR_MESSAGES: 'CLEAR_MESSAGES',
  UPDATE_PAGE_CONTEXT: 'UPDATE_PAGE_CONTEXT',
};

// Reducer function
const chatReducer = (state, action) => {
  switch (action.type) {
    case actionTypes.SET_SESSION_ID:
      return {
        ...state,
        sessionId: action.payload,
      };
    case actionTypes.ADD_MESSAGE:
      return {
        ...state,
        messages: [...state.messages, action.payload],
      };
    case actionTypes.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload,
      };
    case actionTypes.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        isLoading: false,
      };
    case actionTypes.SET_PAGE_CONTEXT:
      return {
        ...state,
        pageContext: action.payload,
      };
    case actionTypes.SET_VISIBILITY:
      return {
        ...state,
        isVisible: action.payload,
      };
    case actionTypes.SET_MINIMIZED:
      return {
        ...state,
        isMinimized: action.payload,
      };
    case actionTypes.CLEAR_MESSAGES:
      return {
        ...state,
        messages: [],
      };
    case actionTypes.UPDATE_PAGE_CONTEXT:
      // Only update if the page context has actually changed
      if (state.pageContext !== action.payload) {
        return {
          ...state,
          pageContext: action.payload,
          // Optionally clear messages when page context changes, or keep them
          // messages: [], // Uncomment if you want to clear messages on page change
        };
      }
      return state;
    default:
      return state;
  }
};

// Create the context
const ChatContext = createContext();

// Provider component
export const ChatProvider = ({ children, initialPageContext = '' }) => {
  const [state, dispatch] = useReducer(chatReducer, {
    ...initialState,
    pageContext: initialPageContext,
  });

  // Load session state from localStorage on initial load
  useEffect(() => {
    const savedSession = localStorage.getItem('chatSession');
    if (savedSession) {
      const sessionData = JSON.parse(savedSession);
      if (sessionData.sessionId) {
        dispatch({ type: actionTypes.SET_SESSION_ID, payload: sessionData.sessionId });
      }
      if (sessionData.messages) {
        dispatch({ type: actionTypes.CLEAR_MESSAGES });
        sessionData.messages.forEach(message => {
          dispatch({ type: actionTypes.ADD_MESSAGE, payload: message });
        });
      }
    }

    // Load visibility and minimized state
    const savedVisibility = localStorage.getItem('chatVisibility');
    if (savedVisibility) {
      const visibilityData = JSON.parse(savedVisibility);
      dispatch({ type: actionTypes.SET_VISIBILITY, payload: visibilityData.isVisible });
      dispatch({ type: actionTypes.SET_MINIMIZED, payload: visibilityData.isMinimized });
    }
  }, []);

  // Save session state to localStorage whenever it changes
  useEffect(() => {
    const sessionData = {
      sessionId: state.sessionId,
      messages: state.messages,
    };
    localStorage.setItem('chatSession', JSON.stringify(sessionData));
  }, [state.sessionId, state.messages]);

  // Save visibility state to localStorage whenever it changes
  useEffect(() => {
    const visibilityData = {
      isVisible: state.isVisible,
      isMinimized: state.isMinimized,
    };
    localStorage.setItem('chatVisibility', JSON.stringify(visibilityData));
  }, [state.isVisible, state.isMinimized]);

  // Function to update page context (useful when user navigates to a different page)
  const updatePageContext = (newPageContext) => {
    dispatch({ type: actionTypes.UPDATE_PAGE_CONTEXT, payload: newPageContext });
  };

  // Function to set session ID
  const setSessionId = (sessionId) => {
    dispatch({ type: actionTypes.SET_SESSION_ID, payload: sessionId });
  };

  // Function to add a message
  const addMessage = (message) => {
    dispatch({ type: actionTypes.ADD_MESSAGE, payload: message });
  };

  // Function to set loading state
  const setLoading = (isLoading) => {
    dispatch({ type: actionTypes.SET_LOADING, payload: isLoading });
  };

  // Function to set error
  const setError = (error) => {
    dispatch({ type: actionTypes.SET_ERROR, payload: error });
  };

  // Function to set page context
  const setPageContext = (pageContext) => {
    dispatch({ type: actionTypes.SET_PAGE_CONTEXT, payload: pageContext });
  };

  // Function to set visibility
  const setVisibility = (isVisible) => {
    dispatch({ type: actionTypes.SET_VISIBILITY, payload: isVisible });
  };

  // Function to set minimized state
  const setMinimized = (isMinimized) => {
    dispatch({ type: actionTypes.SET_MINIMIZED, payload: isMinimized });
  };

  // Function to clear messages
  const clearMessages = () => {
    dispatch({ type: actionTypes.CLEAR_MESSAGES });
  };

  const value = {
    ...state,
    setSessionId,
    addMessage,
    setLoading,
    setError,
    setPageContext,
    setVisibility,
    setMinimized,
    clearMessages,
    updatePageContext,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

// Custom hook to use the chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

export default ChatContext;