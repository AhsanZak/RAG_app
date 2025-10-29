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

// Embedding Models API
export const embeddingModelsAPI = {
  // Get all embedding models
  getAll: async () => {
    try {
      const response = await api.get('/api/embedding-models');
      // Handle both array response and object with models array
      if (Array.isArray(response.data)) {
        return response.data;
      } else if (response.data.models && Array.isArray(response.data.models)) {
        return response.data.models;
      } else {
        console.warn('Unexpected response format:', response.data);
        return [];
      }
    } catch (error) {
      console.error('Error fetching embedding models:', error);
      throw error;
    }
  },

  // Get single model by name
  getByName: async (modelName) => {
    try {
      const response = await api.get(`/api/embedding-models/by-name/${encodeURIComponent(modelName)}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching embedding model:', error);
      throw error;
    }
  },

  // Download model by name
  download: async (modelName) => {
    try {
      const response = await api.post(`/api/embedding-models/${encodeURIComponent(modelName)}/download`);
      return response.data;
    } catch (error) {
      console.error('Error downloading embedding model:', error);
      throw error;
    }
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
  processPDFs: async (files, sessionName, llmModelId, embeddingModelId, embeddingModelName) => {
    const payload = {
      files: files,
      user_id: 1,
      session_name: sessionName,
      llm_model_id: llmModelId || 1,
    };
    
    // Include embedding model ID or name
    if (embeddingModelId) {
      payload.embedding_model_id = embeddingModelId;
    } else if (embeddingModelName) {
      payload.embedding_model_name = embeddingModelName;
    }
    
    const response = await api.post('/api/pdf/process', payload);
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
