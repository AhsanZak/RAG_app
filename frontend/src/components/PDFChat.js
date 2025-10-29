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
  Input
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
import { llmModelsAPI } from '../services/api';
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
    // TODO: Load sessions from backend
    // For now, use sample data
    const sampleSessions = [
      {
        id: 1,
        name: 'Research Papers',
        fileCount: 3,
        createdAt: new Date(Date.now() - 86400000),
        status: 'ready'
      },
      {
        id: 2,
        name: 'Annual Report Analysis',
        fileCount: 1,
        createdAt: new Date(Date.now() - 3600000),
        status: 'processing'
      }
    ];
    setSessions(sampleSessions);
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

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        const newProgress = { ...prev };
        newProgress[file.uid] = (newProgress[file.uid] || 0) + 10;
        if (newProgress[file.uid] >= 100) {
          clearInterval(progressInterval);
          setFiles(prevFiles =>
            prevFiles.map(f =>
              f.uid === file.uid ? { ...f, status: 'uploaded', progress: 100 } : f
            )
          );
          setUploading(false);
        }
        return newProgress;
      });
    }, 200);

    // TODO: Actual file upload to backend
    // const formData = new FormData();
    // formData.append('file', file);
    // const response = await uploadPDF(formData);

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
    if (files.length === 0) return;

    setProcessingFiles(true);
    try {
      // TODO: Process files and create ChromaDB collection
      // const response = await processPDFs(files, selectedModel);
      
      // Simulate processing
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Create new session
      const newSession = {
        id: Date.now(),
        name: files.length === 1 ? files[0].name : `${files.length} PDFs`,
        fileCount: files.length,
        createdAt: new Date(),
        status: 'ready',
        files: files.map(f => f.name)
      };

      setSessions(prev => [newSession, ...prev]);
      setCurrentSession(newSession);
      setFiles([]);
      setUploadProgress({});

      Modal.success({
        title: 'Files Processed Successfully',
        content: 'Your PDF files have been processed and vectorized. You can now start chatting!',
      });
    } catch (error) {
      Modal.error({
        title: 'Processing Failed',
        content: 'Failed to process PDF files. Please try again.',
      });
    } finally {
      setProcessingFiles(false);
    }
  };

  const handleSelectSession = (session) => {
    setCurrentSession(session);
    setChatMessages([]);
  };

  const handleNewSession = () => {
    setCurrentSession(null);
    setFiles([]);
    setChatMessages([]);
    setUploadProgress({});
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !currentSession) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setInputValue('');

    // TODO: Send to backend for RAG processing
    // const response = await sendChatMessage({
    //   message: inputValue,
    //   sessionId: currentSession.id,
    //   modelId: selectedModel
    // });

    // Simulate assistant response
    setTimeout(() => {
      const assistantResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `I understand you're asking about your PDF: "${inputValue}"\n\nBased on the documents you've uploaded, I can help you with that. This is a simulated response. The actual implementation will query ChromaDB for relevant context and generate responses using your selected LLM model.`,
        timestamp: new Date(),
        sources: ['document1.pdf', 'document2.pdf']
      };
      setChatMessages(prev => [...prev, assistantResponse]);
    }, 1000);

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
                      description={`${session.fileCount} file(s)`}
                    />
                    {session.status === 'processing' && (
                      <LoadingOutlined />
                    )}
                    {session.status === 'ready' && (
                      <Tag color="green">Ready</Tag>
                    )}
                  </List.Item>
                )}
              />
            </Card>

            {!currentSession && (
              <Card title="Upload PDF Files" style={{ marginTop: 16 }}>
                <Dragger {...uploadProps}>
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
                      style={{ marginTop: 16 }}
                    >
                      Process & Vectorize PDFs
                    </Button>
                  </div>
                )}
              </Card>
            )}
          </div>

          {/* Right Panel - Chat Area */}
          <div className="pdf-chat-main">
            {!currentSession ? (
              <div className="empty-chat-state">
                <FilePdfOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                <Title level={3}>No Session Selected</Title>
                <Paragraph>
                  Upload PDF files and create a session to start chatting, or select an existing session.
                </Paragraph>
              </div>
            ) : (
              <>
                <div className="chat-messages-container">
                  {chatMessages.length === 0 ? (
                    <div className="empty-messages-state">
                      <RobotOutlined style={{ fontSize: '48px', color: '#d9d9d9', marginBottom: 16 }} />
                      <Title level={4}>Start Chatting</Title>
                      <Paragraph>
                        Ask questions about your PDF documents. The AI will use the vectorized content
                        from ChromaDB to provide accurate answers.
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
                      placeholder="Ask questions about your PDF documents..."
                      autoSize={{ minRows: 1, maxRows: 4 }}
                      className="chat-input"
                    />
                    <Button
                      type="primary"
                      icon={<SendOutlined />}
                      onClick={handleSendMessage}
                      disabled={!inputValue.trim()}
                      className="send-btn"
                    >
                      Send
                    </Button>
                  </div>
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

