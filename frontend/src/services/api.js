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

export default api;
