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

// Database Chat API
export const databaseChatAPI = {
  // Create database connection
  createConnection: async (connectionData) => {
    const response = await api.post('/api/database/connections', connectionData);
    return response.data;
  },

  // Get all connections
  getConnections: async (userId = 1) => {
    const response = await api.get(`/api/database/connections?user_id=${userId}`);
    return response.data;
  },

  // Get single connection
  getConnection: async (connectionId) => {
    const response = await api.get(`/api/database/connections/${connectionId}`);
    return response.data;
  },

  // Update connection
  updateConnection: async (connectionId, connectionData) => {
    const response = await api.put(`/api/database/connections/${connectionId}`, connectionData);
    return response.data;
  },

  // Delete connection
  deleteConnection: async (connectionId) => {
    const response = await api.delete(`/api/database/connections/${connectionId}`);
    return response.data;
  },

  // Test connection
  testConnection: async (connectionData) => {
    const response = await api.post('/api/database/test-connection', connectionData);
    return response.data;
  },

  // Extract schema
  extractSchema: async (connectionData) => {
    const response = await api.post('/api/database/extract-schema', connectionData);
    return response.data;
  },

  // Save schema
  saveSchema: async (schemaData) => {
    const response = await api.post('/api/database/save-schema', schemaData);
    return response.data;
  },

  // Process schema (create embeddings)
  processSchema: async (sessionData) => {
    const response = await api.post('/api/database/process-schema', sessionData);
    return response.data;
  },

  // Chat with database schema
  chat: async (sessionId, message, modelId) => {
    const response = await api.post('/api/database/chat', {
      session_id: sessionId,
      message: message,
      model_id: modelId,
      user_id: 1
    });
    return response.data;
  },

  // Get saved schema for connection
  getSchema: async (connectionId) => {
    const response = await api.get(`/api/database/schemas/${connectionId}`);
    return response.data;
  },

  // Get session messages (reuse from pdfChatAPI)
  getSessionMessages: async (sessionId) => {
    const response = await api.get(`/api/sessions/${sessionId}/messages`);
    return response.data;
  },

  // Get all sessions (optionally filtered by connection_id)
  getSessions: async (userId = 1, connectionId = null) => {
    let url = `/api/sessions?user_id=${userId}`;
    if (connectionId) {
      url += `&connection_id=${connectionId}`;
    }
    const response = await api.get(url);
    return response.data;
  },

  createSession: async ({ connectionId, sessionName, userId = 1, llmModelId, embeddingModelName }) => {
    const payload = {
      connection_id: connectionId,
      session_name: sessionName,
      user_id: userId,
      llm_model_id: llmModelId,
      embedding_model_name: embeddingModelName,
    };
    const response = await api.post('/api/database/sessions', payload);
    return response.data;
  },

  uploadKnowledge: async ({ connectionId, sessionId, title, description, file }) => {
    const formData = new FormData();
    if (connectionId !== undefined && connectionId !== null) {
      formData.append('connection_id', String(connectionId));
    }
    if (sessionId !== undefined && sessionId !== null) {
      formData.append('session_id', String(sessionId));
    }
    if (title) {
      formData.append('title', title);
    }
    if (description) {
      formData.append('description', description);
    }
    formData.append('file', file);
    const response = await api.post('/api/database/knowledge', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  getKnowledge: async (connectionId, sessionId = null) => {
    let url = '/api/database/knowledge';
    const params = [];
    if (connectionId !== undefined && connectionId !== null) {
      params.push(`connection_id=${connectionId}`);
    }
    if (sessionId !== undefined && sessionId !== null) {
      params.push(`session_id=${sessionId}`);
    }
    if (params.length) {
      url += `?${params.join('&')}`;
    }
    const response = await api.get(url);
    return response.data;
  },

  deleteKnowledge: async (knowledgeId) => {
    const response = await api.delete(`/api/database/knowledge/${knowledgeId}`);
    return response.data;
  },

  getKnowledgeDownloadUrl: (knowledgeId) =>
    `${API_BASE_URL}/api/database/knowledge/${knowledgeId}/download`,
};

export default api;
