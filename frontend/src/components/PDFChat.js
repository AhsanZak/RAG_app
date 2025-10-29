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
  Tooltip,
  Modal,
  Input,
  message
} from 'antd';
import {
  UploadOutlined,
  FilePdfOutlined,
  DeleteOutlined,
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  CloseOutlined,
  CheckCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';
import Markdown from 'react-markdown';
import { llmModelsAPI, pdfChatAPI } from '../services/api';
import '../styles/PDFChat.css';

const { Header, Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { Dragger } = Upload;

const PDFChat = ({ onBack }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({});
  const [chatMessages, setChatMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [processingFiles, setProcessingFiles] = useState(false);
  const [sessionReady, setSessionReady] = useState(false); // Track if session has processed PDFs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    loadModels();
    loadSessions();
  }, []);

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

  const loadSessions = async () => {
    try {
      const sessionsData = await pdfChatAPI.getSessions();
      const formattedSessions = sessionsData.map(s => ({
        id: s.id,
        name: s.name,
        fileCount: s.message_count || 0,
        createdAt: new Date(s.created_at),
        status: 'ready',
        hasVectorizedData: true // Sessions from backend have processed PDFs
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
    setUploading(true);

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
      
      message.success(`PDF "${file.name}" uploaded successfully`);
    } catch (error) {
      console.error('Upload error:', error);
      message.error(`Failed to upload ${file.name}`);
      setFiles(prev => prev.filter(f => f.uid !== file.uid));
    } finally {
      setUploading(false);
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
      const response = await pdfChatAPI.processPDFs(filesData, sessionName, selectedModel);

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
          <Select
            value={selectedModel}
            onChange={setSelectedModel}
            style={{ width: 200 }}
            placeholder="Select Model"
          >
            {availableModels.map((model) => (
              <Option key={model.id} value={model.id}>
                {model.model_name} ({model.provider})
              </Option>
            ))}
          </Select>
        </Space>
      </Header>

      <Content className="pdf-chat-content">
        <div className="pdf-chat-container">
          {/* Left Panel - File Upload & Sessions */}
          <div className="pdf-chat-sidebar">
            <Card title="Sessions" extra={<Button type="link" onClick={handleNewSession}>New</Button>}>
              <List
                dataSource={sessions}
                renderItem={(session) => (
                  <List.Item
                    className={`session-item ${currentSession?.id === session.id ? 'active' : ''}`}
                    onClick={() => handleSelectSession(session)}
                  >
                  <List.Item.Meta
                    avatar={<FilePdfOutlined />}
                    title={session.name}
                    description={
                      session.hasVectorizedData 
                        ? `${session.fileCount} file(s) - Ready`
                        : 'New session - Upload PDFs'
                    }
                  />
                  {session.status === 'processing' && (
                    <LoadingOutlined />
                  )}
                  {session.hasVectorizedData && session.status === 'ready' && (
                    <Tag color="green">Ready</Tag>
                  )}
                  {!session.hasVectorizedData && (
                    <Tag color="orange">New</Tag>
                  )}
                  </List.Item>
                )}
              />
            </Card>

            {/* Upload PDF Files - Always visible on left */}
            <Card 
              title={
                <Space>
                  <FilePdfOutlined />
                  <span>Upload & Process PDF Files</span>
                </Space>
              }
              style={{ marginTop: 16 }}
            >
              {!sessionReady && (
                <Alert
                  message="Upload PDF files and process them to enable chat"
                  description="PDFs will be vectorized and stored in ChromaDB for RAG-based chat"
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              )}
              
              <Dragger {...uploadProps} disabled={processingFiles}>
                <p className="ant-upload-drag-icon">
                  <FilePdfOutlined style={{ fontSize: '48px', color: '#ff4d4f' }} />
                </p>
                <p className="ant-upload-text">Click or drag PDF files here to upload</p>
                <p className="ant-upload-hint">Support for single or multiple PDF uploads</p>
              </Dragger>

              {files.length > 0 && (
                <div style={{ marginTop: 16 }}>
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
                </div>
              )}
              
              {sessionReady && (
                <Alert
                  message="Session ready for chat"
                  description="PDFs have been processed and vectorized. You can now chat!"
                  type="success"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </Card>
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
    </Layout>
  );
};

export default PDFChat;

