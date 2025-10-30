import React, { useState, useRef, useEffect } from 'react';
import {
  Layout,
  Upload,
  Button,
  Space,
  Typography,
  Card,
  List,
  Tag,
  Progress,
  Alert,
  Divider,
  Select,
  Modal,
  Input,
  message
} from 'antd';
import {
  FilePdfOutlined,
  DeleteOutlined,
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  CheckCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import Markdown from 'react-markdown';
import { llmModelsAPI, pdfChatAPI, embeddingModelsAPI } from '../services/api';
import '../styles/PDFChat.css';

const { Header, Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Dragger } = Upload;

const PDFChat = ({ onBack }) => {
  const [files, setFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [chatMessages, setChatMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState([]);
  const [selectedEmbeddingModelName, setSelectedEmbeddingModelName] = useState('all-MiniLM-L6-v2'); // Default model
  const [processingFiles, setProcessingFiles] = useState(false);
  const [sessionReady, setSessionReady] = useState(false); // Track if session has processed PDFs
  const [showEmbeddingModal, setShowEmbeddingModal] = useState(false);
  const [downloadingModels, setDownloadingModels] = useState({});
  const [modalEmbeddingModels, setModalEmbeddingModels] = useState([]);
  const [loadingModalModels, setLoadingModalModels] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    loadModels();
    loadEmbeddingModels();
    loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // These should only run once on mount

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const loadModels = async () => {
    try {
      const models = await llmModelsAPI.getAll();
      const activeModels = models.filter(model => model.is_active === 1);
      setAvailableModels(activeModels);
      if (activeModels.length > 0 && !selectedModel) {
        setSelectedModel(activeModels[0].id);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const loadEmbeddingModels = async () => {
    try {
      console.log('Loading embedding models...');
      const models = await embeddingModelsAPI.getAll();
      console.log('Received embedding models:', models);
      
      if (!models || !Array.isArray(models)) {
        console.warn('Invalid models response:', models);
        setAvailableEmbeddingModels([]);
        setSelectedEmbeddingModelName('all-MiniLM-L6-v2');
        return;
      }
      
      const activeModels = models.filter(model => model.is_active === 1 || model.is_active === undefined);
      console.log('Active embedding models:', activeModels);
      setAvailableEmbeddingModels(activeModels);
      
      if (activeModels.length > 0) {
        // Prefer multilingual or English models
        const preferred = activeModels.find(m => 
          m.language === 'multilingual' || m.language === 'english'
        ) || activeModels[0];
        setSelectedEmbeddingModelName(preferred.model_name);
        console.log('Selected default embedding model:', preferred.model_name);
      } else {
        // No models available, use default
        console.warn('No active embedding models found, using default');
        setSelectedEmbeddingModelName('all-MiniLM-L6-v2');
      }
    } catch (error) {
      console.error('Error loading embedding models:', error);
      console.error('Error details:', error.response?.data || error.message);
      // On error, use default
      setSelectedEmbeddingModelName('all-MiniLM-L6-v2');
      setAvailableEmbeddingModels([]);
    }
  };

  const loadSessions = async () => {
    try {
      const sessionsData = await pdfChatAPI.getSessions();
      const formattedSessions = sessionsData.map(s => ({
        id: s.id,
        name: s.name,
        fileCount: s.message_count || 0,
        createdAt: new Date(s.created_at),
        status: s.hasVectorizedData ? 'ready' : 'new',
        hasVectorizedData: s.hasVectorizedData || false // Use backend value
      }));
      
      if (formattedSessions.length === 0) {
        // Auto-create a new session if none exist
        await createNewSession();
      } else {
        setSessions(formattedSessions);
        // Auto-select first session if available
        if (!currentSession && formattedSessions.length > 0) {
          await handleSelectSession(formattedSessions[0]);
        }
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
      // Even on error, create a new session
      await createNewSession();
    }
  };

  const createNewSession = async () => {
    // Create a placeholder session (will be replaced when PDFs are processed)
    const newSession = {
      id: null, // Will be set after processing
      name: 'New Session',
      fileCount: 0,
      createdAt: new Date(),
      status: 'new',
      hasVectorizedData: false
    };
    setSessions([newSession]);
    setCurrentSession(newSession);
    setSessionReady(false);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleFileUpload = async (file) => {
    const newFile = {
      uid: file.uid,
      name: file.name,
      size: file.size,
      status: 'uploading',
      progress: 0
    };

    setFiles(prev => [...prev, newFile]);

    try {
      // Upload PDF to backend
      const response = await pdfChatAPI.uploadPDF(file);
      
      // Update file with extracted chunks
      setFiles(prevFiles =>
        prevFiles.map(f =>
          f.uid === file.uid 
            ? { ...f, status: 'uploaded', progress: 100, chunks: response.chunks, filename: response.filename }
            : f
        )
      );
      
      // Inform user if scanned/image-based PDF detected
      if (response.pdf_type === 'image_pdf') {
        message.info('Scanned PDF detected. Using OCR to extract text. Processing may take longer.');
      }

      message.success(`PDF "${file.name}" uploaded successfully`);
    } catch (error) {
      console.error('Upload error:', error);
      message.error(`Failed to upload ${file.name}`);
      setFiles(prev => prev.filter(f => f.uid !== file.uid));
    }

    return false; // Prevent default upload
  };

  const handleRemoveFile = (file) => {
    setFiles(prev => prev.filter(f => f.uid !== file.uid));
    setUploadProgress(prev => {
      const newProgress = { ...prev };
      delete newProgress[file.uid];
      return newProgress;
    });
  };

  const handleProcessFiles = async () => {
    if (files.length === 0) {
      message.warning('Please upload PDF files first');
      return;
    }

    if (!selectedModel) {
      message.warning('Please select an LLM model first');
      return;
    }

    // Use default embedding model if none selected
    const embeddingModelName = selectedEmbeddingModelName || 'all-MiniLM-L6-v2';

    setProcessingFiles(true);
    try {
      // Prepare files data
      const filesData = files.map(f => ({
        filename: f.filename || f.name,
        chunks: f.chunks || []
      }));

      // Process files and create session
      const sessionName = files.length === 1 
        ? files[0].name.replace('.pdf', '') 
        : `${files.length} PDFs`;
      const response = await pdfChatAPI.processPDFs(
        filesData, 
        sessionName, 
        selectedModel, 
        null,
        embeddingModelName
      );

      // Update session with real data
      const updatedSession = {
        id: response.session_id,
        name: sessionName,
        fileCount: files.length,
        createdAt: new Date(),
        status: 'ready',
        hasVectorizedData: true,
        files: files.map(f => f.filename || f.name)
      };

      // Update sessions list
      setSessions(prev => {
        const updated = prev.map(s => 
          s.id === currentSession?.id ? updatedSession : s
        );
        // If session didn't exist, add it
        if (!updated.find(s => s.id === updatedSession.id)) {
          return [updatedSession, ...updated];
        }
        return updated;
      });

      setCurrentSession(updatedSession);
      
      // Inject preview/summary as the first assistant message for verification
      if (response.preview && (response.preview.summary || (response.preview.samples && response.preview.samples.length))) {
        const previewText = response.preview.summary || response.preview.samples.join('\n\n');
        const assistantPreview = {
          id: response.assistant_message_id || `preview_${Date.now()}`,
          role: 'assistant',
          content: previewText,
          timestamp: new Date()
        };
        setChatMessages([assistantPreview]);
      }
      setSessionReady(true); // Enable chat after processing
      setFiles([]);
      setUploadProgress({});

      Modal.success({
        title: 'PDFs Processed Successfully',
        content: `${response.total_documents} document chunks have been vectorized and stored in ChromaDB. You can now start chatting!`,
      });
      
      // Load messages for the new session
      await handleSelectSession(updatedSession);
    } catch (error) {
      console.error('Processing error:', error);
      Modal.error({
        title: 'Processing Failed',
        content: error.response?.data?.detail || 'Failed to process PDF files. Please try again.',
      });
    } finally {
      setProcessingFiles(false);
    }
  };

  const handleSelectSession = async (session) => {
    setCurrentSession(session);
    setChatMessages([]);
    
    // Check if session has vectorized data
    setSessionReady(session.hasVectorizedData === true && session.id !== null);
    
    if (session.id && session.hasVectorizedData) {
      // Load session messages
      try {
        const messages = await pdfChatAPI.getSessionMessages(session.id);
        const formattedMessages = messages.map(m => ({
          id: m.id,
          role: m.role,
          content: m.message,
          timestamp: new Date(m.created_at),
          sources: m.metadata?.sources || []
        }));
        setChatMessages(formattedMessages);
      } catch (error) {
        console.error('Error loading messages:', error);
        message.error('Failed to load chat history');
      }
    }
  };

  const handleNewSession = () => {
    createNewSession();
    setFiles([]);
    setChatMessages([]);
    setUploadProgress({});
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !currentSession || !selectedModel) return;
    
    if (!sessionReady) {
      message.warning('Please process PDF files first before chatting');
      return;
    }

    if (!currentSession.id) {
      message.warning('Session not ready. Please process PDF files first.');
      return;
    }

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    const messageText = inputValue;
    setChatMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      // Send to backend for RAG processing
      const response = await pdfChatAPI.chat(
        currentSession.id,
        messageText,
        selectedModel
      );

      const assistantResponse = {
        id: response.message_id || Date.now() + 1,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        sources: response.sources || []
      };
      
      setChatMessages(prev => [...prev, assistantResponse]);
      
      // Reload sessions to update message count
      loadSessions();
    } catch (error) {
      console.error('Chat error:', error);
      message.error(error.response?.data?.detail || 'Failed to send message');
      
      // Remove user message on error
      setChatMessages(prev => prev.filter(m => m.id !== userMessage.id));
    }

    inputRef.current?.focus();
  };

  const handleDownloadEmbeddingModel = async (modelName) => {
    setDownloadingModels(prev => ({ ...prev, [modelName]: true }));
    try {
      const response = await embeddingModelsAPI.download(modelName);
      if (response.success) {
        message.success(`Model "${modelName}" downloaded successfully`);
        // Reload models to refresh the list
        await loadEmbeddingModels();
        // Also reload modal models
        const models = await embeddingModelsAPI.getAll();
        setModalEmbeddingModels(models || []);
      }
    } catch (error) {
      console.error('Download error:', error);
      message.error(`Failed to download model: ${error.response?.data?.detail || error.message}`);
    } finally {
      setDownloadingModels(prev => {
        const updated = { ...prev };
        delete updated[modelName];
        return updated;
      });
    }
  };

  const handleManageEmbeddingModels = async () => {
    // Load models specifically for the modal
    setLoadingModalModels(true);
    try {
      console.log('Loading embedding models for modal...');
      const models = await embeddingModelsAPI.getAll();
      console.log('Loaded embedding models for modal:', models);
      
      if (!models || !Array.isArray(models)) {
        console.warn('Invalid models response in modal:', models);
        message.warning('Failed to load embedding models. Invalid response format.');
        setModalEmbeddingModels([]);
      } else {
        // Don't filter by is_active for modal - show all available models
        setModalEmbeddingModels(models);
        
        if (models.length === 0) {
          console.warn('No embedding models returned from API');
          message.warning('No embedding models found. Please check backend configuration.');
        }
      }
    } catch (error) {
      console.error('Error loading embedding models for modal:', error);
      console.error('Error response:', error.response?.data);
      const errorMessage = error.response?.data?.detail || error.message || 'Check your connection';
      message.error(`Failed to load embedding models: ${errorMessage}`);
      setModalEmbeddingModels([]);
    } finally {
      setLoadingModalModels(false);
    }
    setShowEmbeddingModal(true);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const uploadProps = {
    accept: '.pdf',
    multiple: true,
    beforeUpload: handleFileUpload,
    showUploadList: false,
    fileList: files
  };

  return (
    <Layout className="pdf-chat-layout">
      <Header className="pdf-chat-header">
        <Space>
          <Button type="text" onClick={onBack}>
            ← Back
          </Button>
          <Divider type="vertical" />
          <FilePdfOutlined style={{ fontSize: '20px', color: '#ff4d4f' }} />
          <Title level={4} style={{ margin: 0 }}>PDF Chat</Title>
        </Space>
        <Space>
          <Text strong>LLM Model:</Text>
          <Select
            value={selectedModel}
            onChange={setSelectedModel}
            style={{ width: 200 }}
            placeholder="Select LLM Model"
          >
            {availableModels.map((model) => (
              <Option key={model.id} value={model.id}>
                {model.model_name} ({model.provider})
              </Option>
            ))}
          </Select>
          <Divider type="vertical" />
          <Text strong>Embedding:</Text>
          {availableEmbeddingModels.length > 0 ? (
            <Space>
              <Select
                value={selectedEmbeddingModelName}
                onChange={(value) => {
                  setSelectedEmbeddingModelName(value);
                }}
                style={{ width: 250 }}
                placeholder="Select Embedding Model"
              >
                {availableEmbeddingModels.map((model) => (
                  <Option key={model.model_name} value={model.model_name}>
                    {model.display_name} ({model.language})
                  </Option>
                ))}
              </Select>
              <Button 
                type="link" 
                size="small"
                onClick={handleManageEmbeddingModels}
              >
                Manage
              </Button>
            </Space>
          ) : (
            <Space>
              <Tag color="default">{selectedEmbeddingModelName} (Default)</Tag>
              <Button 
                type="link" 
                size="small"
                onClick={handleManageEmbeddingModels}
              >
                Manage Models
              </Button>
            </Space>
          )}
        </Space>
      </Header>

      <Content className="pdf-chat-content">
        <div className="pdf-chat-container">
          {/* Left Panel - Upload (static) + Sessions (scrollable) */}
          <div className="pdf-chat-sidebar">
            {/* Upload - compact and static */}
            <Card 
              className="sidebar-upload"
              size="small"
              title={
                <Space>
                  <FilePdfOutlined />
                  <span>Upload & Process</span>
                </Space>
              }
            >
              {!sessionReady && (
                <Alert
                  message="Upload PDF files and process them to enable chat"
                  description="PDFs will be vectorized and stored in ChromaDB for RAG-based chat"
                  type="info"
                  showIcon
                  style={{ marginBottom: 12 }}
                />
              )}
              
              <Dragger {...uploadProps} disabled={processingFiles}>
                <p className="ant-upload-drag-icon">
                  <FilePdfOutlined style={{ fontSize: '32px', color: '#ff4d4f' }} />
                </p>
                <p className="ant-upload-text" style={{ marginBottom: 4 }}>Click or drag PDFs to upload</p>
                <p className="ant-upload-hint" style={{ marginBottom: 0 }}>Single or multiple files</p>
              </Dragger>

              {files.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <List
                    dataSource={files}
                    renderItem={(file) => (
                      <List.Item
                        actions={[
                          <Button
                            type="text"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => handleRemoveFile(file)}
                            disabled={processingFiles}
                          />
                        ]}
                      >
                        <List.Item.Meta
                          avatar={<FilePdfOutlined />}
                          title={file.name}
                          description={
                            file.status === 'uploading' ? (
                              <Progress
                                percent={uploadProgress[file.uid] || 0}
                                size="small"
                                status="active"
                              />
                            ) : (
                              <Tag color="green">
                                <CheckCircleOutlined /> Uploaded
                              </Tag>
                            )
                          }
                        />
                      </List.Item>
                    )}
                  />
                  <Button
                    type="primary"
                    block
                    loading={processingFiles}
                    onClick={handleProcessFiles}
                    disabled={!selectedModel || processingFiles}
                    style={{ marginTop: 16 }}
                  >
                    {processingFiles ? 'Processing & Vectorizing...' : 'Process & Vectorize PDFs'}
                  </Button>
                  {!selectedModel && (
                    <Alert
                      message="Please select an LLM model from the header first"
                      type="warning"
                      showIcon
                      style={{ marginTop: 8 }}
                    />
                  )}
                  {availableEmbeddingModels.length === 0 && (
                    <Alert
                      message={`Using default embedding model: ${selectedEmbeddingModelName}. Add models in Settings to use different models.`}
                      type="info"
                      showIcon
                      style={{ marginTop: 8 }}
                    />
                  )}
                </div>
              )}
              
              {sessionReady && (
                <Alert
                  message="Session ready for chat"
                  description="PDFs have been processed and vectorized. You can now chat!"
                  type="success"
                  showIcon
                  style={{ marginTop: 12 }}
                />
              )}
            </Card>

            {/* Sessions - scrollable list */}
            <div className="sidebar-sessions">
              <Card title="Sessions" size="small" extra={<Button type="link" onClick={handleNewSession}>New</Button>}>
                <List
                  size="small"
                  dataSource={sessions}
                  renderItem={(session) => (
                    <List.Item
                      className={`session-item ${currentSession?.id === session.id ? 'active' : ''}`}
                      onClick={() => handleSelectSession(session)}
                    >
                      <List.Item.Meta
                        avatar={<FilePdfOutlined />}
                        title={<span style={{ fontSize: 13 }}>{session.name}</span>}
                        description={
                          <span style={{ fontSize: 12 }}>
                            {session.hasVectorizedData ? `${session.fileCount} file(s) - Ready` : 'New session - Upload PDFs'}
                          </span>
                        }
                      />
                      {session.status === 'processing' && (
                        <LoadingOutlined />
                      )}
                      {session.hasVectorizedData && session.status === 'ready' && (
                        <Tag color="green" style={{ marginInlineStart: 6 }}>Ready</Tag>
                      )}
                      {!session.hasVectorizedData && (
                        <Tag color="orange" style={{ marginInlineStart: 6 }}>New</Tag>
                      )}
                    </List.Item>
                  )}
                />
              </Card>
            </div>
          </div>

          {/* Right Panel - Chat Area */}
          <div className="pdf-chat-main">
            {!currentSession ? (
              <div className="empty-chat-state">
                <FilePdfOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                <Title level={3}>Ready to Start</Title>
                <Paragraph>
                  Upload PDF files on the left side, then process them to enable chat.
                </Paragraph>
              </div>
            ) : (
              <>
                <div className="chat-messages-container">
                  {chatMessages.length === 0 ? (
                    <div className="empty-messages-state">
                      <RobotOutlined style={{ fontSize: '48px', color: '#d9d9d9', marginBottom: 16 }} />
                      <Title level={4}>
                        {sessionReady ? 'Start Chatting' : 'Process PDFs First'}
                      </Title>
                      <Paragraph>
                        {sessionReady ? (
                          <>
                            Ask questions about your PDF documents. The AI will use the vectorized content
                            from ChromaDB to provide accurate answers.
                          </>
                        ) : (
                          <>
                            Upload PDF files on the left side and click "Process & Vectorize PDFs" 
                            to enable chat. PDFs will be converted to vectors and stored in ChromaDB.
                          </>
                        )}
                      </Paragraph>
                    </div>
                  ) : (
                    <div className="messages-list">
                      {chatMessages.map((message) => (
                        <div
                          key={message.id}
                          className={`message-item ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
                        >
                          <div className="message-avatar">
                            {message.role === 'user' ? (
                              <UserOutlined />
                            ) : (
                              <RobotOutlined />
                            )}
                          </div>
                          <div className="message-content">
                            <div className="message-bubble">
                              <Markdown>{message.content}</Markdown>
                            </div>
                            {message.sources && message.sources.length > 0 && (
                              <div className="message-sources">
                                <Text type="secondary" style={{ fontSize: '12px' }}>
                                  Sources: {message.sources.join(', ')}
                                </Text>
                              </div>
                            )}
                            <div className="message-timestamp">
                              {message.timestamp.toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      ))}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>

                <Footer className="chat-input-footer">
                  <div className="input-container">
                    <Input.TextArea
                      ref={inputRef}
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={
                        sessionReady 
                          ? "Ask questions about your PDF documents..."
                          : "Please process PDF files first to enable chat..."
                      }
                      autoSize={{ minRows: 1, maxRows: 4 }}
                      className="chat-input"
                      disabled={!sessionReady}
                    />
                    <Button
                      type="primary"
                      icon={<SendOutlined />}
                      onClick={handleSendMessage}
                      disabled={!inputValue.trim() || !sessionReady}
                      className="send-btn"
                    >
                      Send
                    </Button>
                  </div>
                  {!sessionReady && (
                    <Alert
                      message="Chat disabled"
                      description="Please upload and process PDF files first to enable chat"
                      type="warning"
                      showIcon
                      style={{ marginTop: 8 }}
                    />
                  )}
                </Footer>
              </>
            )}
          </div>
        </div>
      </Content>

      {/* Embedding Models Management Modal */}
      <Modal
        title="Manage Embedding Models"
        open={showEmbeddingModal}
        onCancel={() => setShowEmbeddingModal(false)}
        footer={[
          <Button key="close" onClick={() => setShowEmbeddingModal(false)}>
            Close
          </Button>
        ]}
        width={800}
      >
        <Alert
          message="Available Embedding Models"
          description="Select a model from the list below. Models will be downloaded automatically when first used."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
        {loadingModalModels ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <LoadingOutlined style={{ fontSize: '32px' }} />
            <p style={{ marginTop: 16 }}>Loading embedding models...</p>
          </div>
        ) : (
          <>
            <List
              dataSource={modalEmbeddingModels}
              renderItem={(model) => {
                const isDownloading = downloadingModels[model.model_name];
                const isSelected = selectedEmbeddingModelName === model.model_name;
                
                return (
                  <List.Item
                    actions={[
                      isSelected ? (
                        <Tag color="green">Selected</Tag>
                      ) : (
                        <Button
                          type="primary"
                          size="small"
                          onClick={() => {
                            setSelectedEmbeddingModelName(model.model_name);
                            setShowEmbeddingModal(false);
                            message.success(`Selected "${model.display_name}"`);
                            // Also update the dropdown list
                            loadEmbeddingModels();
                          }}
                        >
                          Select
                        </Button>
                      ),
                      <Button
                        type="default"
                        size="small"
                        loading={isDownloading}
                        onClick={() => handleDownloadEmbeddingModel(model.model_name)}
                      >
                        {isDownloading ? 'Downloading...' : 'Download'}
                      </Button>
                    ]}
                  >
                    <List.Item.Meta
                      title={
                        <Space>
                          <Text strong>{model.display_name}</Text>
                          {model.is_default && <Tag color="blue">Default</Tag>}
                          <Tag>{model.language}</Tag>
                          <Tag>{model.provider}</Tag>
                        </Space>
                      }
                      description={
                        <div>
                          <Text type="secondary">{model.description}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            Model: {model.model_name} | Dimension: {model.dimension}
                          </Text>
                        </div>
                      }
                    />
                  </List.Item>
                );
              }}
            />
            
            {modalEmbeddingModels.length === 0 && !loadingModalModels && (
              <Alert
                message="No models available"
                description="Failed to load embedding models. Please check your connection and try again."
                type="warning"
                showIcon
              />
            )}
          </>
        )}
      </Modal>
    </Layout>
  );
};

export default PDFChat;

