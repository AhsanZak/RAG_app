import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// LLM Models API
export const llmModelsAPI = {
  // Get all LLM models
  getAll: async () => {
    const response = await api.get('/api/llm-models');
    return response.data;
  },

  // Get single model by ID
  getById: async (id) => {
    const response = await api.get(`/api/llm-models/${id}`);
    return response.data;
  },

  // Create new model
  create: async (modelData) => {
    const response = await api.post('/api/llm-models', modelData);
    return response.data;
  },

  // Update model
  update: async (id, modelData) => {
    const response = await api.put(`/api/llm-models/${id}`, modelData);
    return response.data;
  },

  // Delete model
  delete: async (id) => {
    const response = await api.delete(`/api/llm-models/${id}`);
    return response.data;
  },

  // Test model connection
  testConnection: async (modelData) => {
    const response = await api.post('/api/llm-models/test', modelData);
    return response.data;
  }
};

// Users API
export const usersAPI = {
  getAll: async () => {
    const response = await api.get('/api/users');
    return response.data;
  },
};

// Health check
export const healthAPI = {
  check: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

// PDF Chat API
export const pdfChatAPI = {
  // Upload PDF file
  uploadPDF: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', '1');
    
    const response = await api.post('/api/pdf/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Process PDFs and create session
  processPDFs: async (files, sessionName, llmModelId) => {
    const response = await api.post('/api/pdf/process', {
      files: files,
      user_id: 1,
      session_name: sessionName,
      llm_model_id: llmModelId || 1
    });
    return response.data;
  },

  // Chat with PDF documents
  chat: async (sessionId, message, modelId) => {
    const response = await api.post('/api/pdf/chat', {
      session_id: sessionId,
      message: message,
      model_id: modelId,
      user_id: 1
    });
    return response.data;
  },

  // Get all sessions
  getSessions: async (userId = 1) => {
    const response = await api.get(`/api/sessions?user_id=${userId}`);
    return response.data;
  },

  // Get session messages
  getSessionMessages: async (sessionId) => {
    const response = await api.get(`/api/sessions/${sessionId}/messages`);
    return response.data;
  },
};

export default api;
