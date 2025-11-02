import React, { useState, useRef, useEffect } from 'react';
import {
  Layout,
  Button,
  Space,
  Typography,
  Card,
  List,
  Tag,
  Alert,
  Divider,
  Select,
  Modal,
  Input,
  Form,
  message,
  Tabs,
  Descriptions
} from 'antd';
import {
  DatabaseOutlined,
  DeleteOutlined,
  SendOutlined,
  RobotOutlined,
  UserOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  PlusOutlined,
  ReloadOutlined,
  SyncOutlined
} from '@ant-design/icons';
import Markdown from 'react-markdown';
import { llmModelsAPI, databaseChatAPI, embeddingModelsAPI } from '../services/api';
import '../styles/PDFChat.css';

const { Header, Content, Footer } = Layout;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

const DatabaseChat = ({ onBack }) => {
  const [connections, setConnections] = useState([]);
  const [currentConnection, setCurrentConnection] = useState(null);
  const [schemaData, setSchemaData] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState([]);
  const [selectedEmbeddingModelName, setSelectedEmbeddingModelName] = useState('all-MiniLM-L6-v2');
  const [processingSchema, setProcessingSchema] = useState(false);
  const [extractingSchema, setExtractingSchema] = useState(false);
  const [sessionReady, setSessionReady] = useState(false);
  const [showConnectionModal, setShowConnectionModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingConnection, setEditingConnection] = useState(null);
  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionForm] = Form.useForm();
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Default ports for different database types
  const defaultPorts = {
    postgresql: 5432,
    mysql: 3306,
    mariadb: 3306,
    sqlite: null, // SQLite doesn't use a port
    oracle: 1521,
    mssql: 1433
  };

  useEffect(() => {
    loadModels();
    loadEmbeddingModels();
    loadConnections();
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const loadEmbeddingModels = async () => {
    try {
      const models = await embeddingModelsAPI.getAll();
      if (models && Array.isArray(models)) {
        const activeModels = models.filter(model => model.is_active === 1 || model.is_active === undefined);
        setAvailableEmbeddingModels(activeModels);
        if (activeModels.length > 0) {
          const preferred = activeModels.find(m => 
            m.language === 'multilingual' || m.language === 'english'
          ) || activeModels[0];
          setSelectedEmbeddingModelName(preferred.model_name);
        }
      }
    } catch (error) {
      console.error('Error loading embedding models:', error);
    }
  };

  const loadConnections = async () => {
    try {
      const connectionsData = await databaseChatAPI.getConnections();
      setConnections(connectionsData);
    } catch (error) {
      console.error('Error loading connections:', error);
      message.error('Failed to load database connections');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleTestConnection = async () => {
    const values = await connectionForm.validateFields();
    setTestingConnection(true);
    try {
      const result = await databaseChatAPI.testConnection(values);
      if (result.success) {
        message.success('Connection test successful!');
      } else {
        message.error(result.message || 'Connection test failed');
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Connection test failed');
    } finally {
      setTestingConnection(false);
    }
  };

  const handleCreateConnection = async () => {
    try {
      const values = await connectionForm.validateFields();
      const connectionData = {
        ...values,
        user_id: 1
      };
      
      const result = await databaseChatAPI.createConnection(connectionData);
      if (result.success) {
        message.success('Connection created successfully');
        setShowConnectionModal(false);
        connectionForm.resetFields();
        await loadConnections();
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to create connection');
    }
  };

  const handleUpdateConnection = async () => {
    try {
      const values = await connectionForm.validateFields();
      const result = await databaseChatAPI.updateConnection(editingConnection.id, values);
      if (result.success) {
        message.success('Connection updated successfully');
        setShowEditModal(false);
        setEditingConnection(null);
        connectionForm.resetFields();
        await loadConnections();
        if (currentConnection && currentConnection.id === editingConnection.id) {
          await handleSelectConnection({ ...editingConnection, ...values });
        }
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to update connection');
    }
  };

  const handleDeleteConnection = async (connection) => {
    Modal.confirm({
      title: 'Delete Connection',
      content: `Are you sure you want to delete connection "${connection.name}"?`,
      onOk: async () => {
        try {
          await databaseChatAPI.deleteConnection(connection.id);
          message.success('Connection deleted successfully');
          await loadConnections();
          if (currentConnection && currentConnection.id === connection.id) {
            setCurrentConnection(null);
            setSchemaData(null);
            setSessionReady(false);
          }
        } catch (error) {
          message.error(error.response?.data?.detail || 'Failed to delete connection');
        }
      }
    });
  };

  const handleSelectConnection = async (connection) => {
    setCurrentConnection(connection);
    setChatMessages([]);
    setSessionReady(false);
    
    // Try to load existing schema
    try {
      const schema = await databaseChatAPI.getSchema(connection.id);
      if (schema && schema.schema_data) {
        setSchemaData(schema);
      } else {
        setSchemaData(null);
      }
    } catch (error) {
      // No schema found, that's okay
      setSchemaData(null);
    }
  };

  const handleExtractSchema = async () => {
    if (!currentConnection) {
      message.warning('Please select a connection first');
      return;
    }

    setExtractingSchema(true);
    try {
      const result = await databaseChatAPI.extractSchema({
        connection_id: currentConnection.id
      });
      
      if (result.success) {
        // Save schema to database
        const saveResult = await databaseChatAPI.saveSchema({
          connection_id: currentConnection.id,
          schema_data: result.schema_data,
          schema_text: result.schema_text
        });
        
        if (saveResult.success) {
          message.success('Schema extracted and saved successfully');
          setSchemaData({
            schema_data: result.schema_data,
            schema_text: result.schema_text,
            connection_id: currentConnection.id
          });
        }
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to extract schema');
    } finally {
      setExtractingSchema(false);
    }
  };

  const handleProcessSchema = async () => {
    if (!schemaData || !currentConnection) {
      message.warning('Please extract schema first');
      return;
    }

    if (!selectedModel) {
      message.warning('Please select an LLM model first');
      return;
    }

    setProcessingSchema(true);
    try {
      const sessionName = `${currentConnection.name} - Schema`;
      const result = await databaseChatAPI.processSchema({
        connection_id: currentConnection.id,
        session_name: sessionName,
        llm_model_id: selectedModel,
        embedding_model_name: selectedEmbeddingModelName || 'all-MiniLM-L6-v2',
        user_id: 1
      });

      if (result.success) {
        // Create session object
        const newSession = {
          id: result.session_id,
          name: sessionName,
          createdAt: new Date(),
          status: 'ready',
          hasVectorizedData: true
        };

        setSessions([newSession]);
        setCurrentSession(newSession);
        setSessionReady(true);
        
        // Inject preview as first assistant message
        if (result.preview && result.preview.summary) {
          const assistantPreview = {
            id: `preview_${Date.now()}`,
            role: 'assistant',
            content: result.preview.summary,
            timestamp: new Date()
          };
          setChatMessages([assistantPreview]);
        }

        Modal.success({
          title: 'Schema Processed Successfully',
          content: `${result.total_chunks} schema chunks have been vectorized. You can now start chatting!`,
        });
      }
    } catch (error) {
      message.error(error.response?.data?.detail || 'Failed to process schema');
    } finally {
      setProcessingSchema(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !currentSession || !selectedModel) return;
    
    if (!sessionReady) {
      message.warning('Please process schema first before chatting');
      return;
    }

    if (!currentSession.id) {
      message.warning('Session not ready. Please process schema first.');
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
      const response = await databaseChatAPI.chat(
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
    } catch (error) {
      console.error('Chat error:', error);
      message.error(error.response?.data?.detail || 'Failed to send message');
      
      // Remove user message on error
      setChatMessages(prev => prev.filter(m => m.id !== userMessage.id));
    }

    inputRef.current?.focus();
  };

  const handleEditConnection = (connection) => {
    setEditingConnection(connection);
    connectionForm.setFieldsValue({
      name: connection.name,
      database_type: connection.database_type,
      host: connection.host,
      port: connection.port,
      database_name: connection.database_name,
      username: connection.username,
      password: '', // Don't show password
      schema_name: connection.schema_name
    });
    setShowEditModal(true);
  };

  const handleDatabaseTypeChange = (databaseType, form) => {
    // Auto-fill default port when database type changes
    const defaultPort = defaultPorts[databaseType];
    if (databaseType === 'sqlite') {
      // Clear port for SQLite since it doesn't use a port
      form.setFieldsValue({ port: null });
    } else if (defaultPort !== null && defaultPort !== undefined) {
      // Always set default port (user can edit it if needed)
      form.setFieldsValue({ port: defaultPort });
    }
  };

  return (
    <Layout className="pdf-chat-layout">
      <Header className="pdf-chat-header">
        <Space>
          <Button type="text" onClick={onBack}>
            ← Back
          </Button>
          <Divider type="vertical" />
          <DatabaseOutlined style={{ fontSize: '20px', color: '#722ed1' }} />
          <Title level={4} style={{ margin: 0 }}>Database Chat</Title>
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
        </Space>
      </Header>

      <Content className="pdf-chat-content">
        <div className="pdf-chat-container">
          {/* Left Panel - Connections & Schema */}
          <div className="pdf-chat-sidebar">
            <Card 
              className="sidebar-upload"
              size="small"
              title={
                <Space>
                  <DatabaseOutlined />
                  <span>Database Connections</span>
                </Space>
              }
              extra={
                <Button 
                  type="primary" 
                  size="small" 
                  icon={<PlusOutlined />}
                  onClick={() => setShowConnectionModal(true)}
                >
                  Add
                </Button>
              }
            >
              {connections.length === 0 ? (
                <Alert
                  message="No connections"
                  description="Add a database connection to get started"
                  type="info"
                  showIcon
                />
              ) : (
                <List
                  size="small"
                  dataSource={connections}
                  renderItem={(connection) => (
                    <List.Item
                      className={`session-item ${currentConnection?.id === connection.id ? 'active' : ''}`}
                      onClick={() => handleSelectConnection(connection)}
                      actions={[
                        <Button
                          type="text"
                          size="small"
                          icon={<ReloadOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleEditConnection(connection);
                          }}
                        />,
                        <Button
                          type="text"
                          danger
                          size="small"
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteConnection(connection);
                          }}
                        />
                      ]}
                    >
                      <List.Item.Meta
                        avatar={<DatabaseOutlined />}
                        title={<span style={{ fontSize: 13 }}>{connection.name}</span>}
                        description={
                          <span style={{ fontSize: 12 }}>
                            {connection.database_type} - {connection.database_name}
                          </span>
                        }
                      />
                    </List.Item>
                  )}
                />
              )}
            </Card>

            {currentConnection && (
              <Card
                className="sidebar-sessions"
                size="small"
                title="Schema Management"
                style={{ marginTop: 12 }}
              >
                <Space direction="vertical" style={{ width: '100%' }}>
                  {!schemaData ? (
                    <>
                      <Alert
                        message="Schema not extracted"
                        description="Click 'Extract Schema' to extract database schema"
                        type="info"
                        showIcon
                        style={{ marginBottom: 12 }}
                      />
                      <Button
                        type="primary"
                        block
                        loading={extractingSchema}
                        icon={<SyncOutlined />}
                        onClick={handleExtractSchema}
                      >
                        Extract Schema
                      </Button>
                    </>
                  ) : (
                    <>
                      <Alert
                        message="Schema extracted"
                        description={`${schemaData.schema_data?.metadata?.total_tables || 0} tables, ${schemaData.schema_data?.metadata?.total_columns || 0} columns`}
                        type="success"
                        showIcon
                        style={{ marginBottom: 12 }}
                      />
                      <Button
                        block
                        icon={<SyncOutlined />}
                        onClick={handleExtractSchema}
                        loading={extractingSchema}
                      >
                        Re-extract Schema
                      </Button>
                      {!sessionReady && (
                        <Button
                          type="primary"
                          block
                          loading={processingSchema}
                          onClick={handleProcessSchema}
                          disabled={!selectedModel || processingSchema}
                          style={{ marginTop: 8 }}
                        >
                          Process & Vectorize Schema
                        </Button>
                      )}
                      {sessionReady && (
                        <Alert
                          message="Schema processed"
                          description="Schema has been vectorized. You can now chat!"
                          type="success"
                          showIcon
                          style={{ marginTop: 8 }}
                        />
                      )}
                    </>
                  )}
                </Space>
              </Card>
            )}
          </div>

          {/* Right Panel - Chat Area */}
          <div className="pdf-chat-main">
            {!currentConnection ? (
              <div className="empty-chat-state">
                <DatabaseOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                <Title level={3}>Select a Connection</Title>
                <Paragraph>
                  Select a database connection from the left panel to start chatting with your database schema.
                </Paragraph>
              </div>
            ) : !schemaData ? (
              <div className="empty-chat-state">
                <SyncOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                <Title level={3}>Extract Schema</Title>
                <Paragraph>
                  Click "Extract Schema" in the left panel to extract and analyze your database schema.
                </Paragraph>
              </div>
            ) : !sessionReady ? (
              <div className="empty-chat-state">
                <RobotOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                <Title level={3}>Process Schema</Title>
                <Paragraph>
                  Click "Process & Vectorize Schema" in the left panel to enable chat functionality.
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
                        Ask questions about your database schema. The AI will use the vectorized schema
                        information to provide accurate answers.
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
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendMessage();
                        }
                      }}
                      placeholder="Ask questions about your database schema..."
                      autoSize={{ minRows: 1, maxRows: 4 }}
                      className="chat-input"
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
                </Footer>
              </>
            )}
          </div>
        </div>
      </Content>

      {/* Connection Modal */}
      <Modal
        title="Add Database Connection"
        open={showConnectionModal}
        onCancel={() => {
          setShowConnectionModal(false);
          connectionForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={connectionForm}
          layout="vertical"
          onFinish={handleCreateConnection}
        >
          <Form.Item
            name="name"
            label="Connection Name"
            rules={[{ required: true, message: 'Please enter connection name' }]}
          >
            <Input placeholder="My Database" />
          </Form.Item>
          
          <Form.Item
            name="database_type"
            label="Database Type"
            rules={[{ required: true, message: 'Please select database type' }]}
          >
            <Select 
              placeholder="Select database type"
              onChange={(value) => handleDatabaseTypeChange(value, connectionForm)}
            >
              <Option value="postgresql">PostgreSQL (Default: 5432)</Option>
              <Option value="mysql">MySQL (Default: 3306)</Option>
              <Option value="mariadb">MariaDB (Default: 3306)</Option>
              <Option value="sqlite">SQLite (No port needed)</Option>
              <Option value="oracle">Oracle (Default: 1521)</Option>
              <Option value="mssql">Microsoft SQL Server (Default: 1433)</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="host"
            label="Host"
            rules={[{ required: true, message: 'Please enter host' }]}
          >
            <Input placeholder="localhost" />
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => 
              prevValues.database_type !== currentValues.database_type
            }
          >
            {({ getFieldValue }) => {
              const dbType = getFieldValue('database_type');
              const defaultPort = defaultPorts[dbType];
              
              if (dbType === 'sqlite') {
                return (
                  <Form.Item name="port" label="Port">
                    <Alert
                      message="SQLite doesn't require a port"
                      type="info"
                      showIcon
                      style={{ marginBottom: 0 }}
                    />
                  </Form.Item>
                );
              }
              
              const placeholder = defaultPort 
                ? `Default: ${defaultPort}` 
                : 'Enter port number';
              
              return (
                <Form.Item name="port" label="Port">
                  <Input 
                    type="number" 
                    placeholder={placeholder}
                  />
                </Form.Item>
              );
            }}
          </Form.Item>

          <Form.Item
            name="database_name"
            label="Database Name"
            rules={[{ required: true, message: 'Please enter database name' }]}
          >
            <Input placeholder="mydb" />
          </Form.Item>

          <Form.Item
            name="username"
            label="Username"
            rules={[{ required: true, message: 'Please enter username' }]}
          >
            <Input placeholder="user" />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password"
            rules={[{ required: true, message: 'Please enter password' }]}
          >
            <Input.Password placeholder="password" />
          </Form.Item>

          <Form.Item
            name="schema_name"
            label="Schema Name (optional)"
          >
            <Input placeholder="public" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button onClick={() => setShowConnectionModal(false)}>
                Cancel
              </Button>
              <Button onClick={handleTestConnection} loading={testingConnection}>
                Test Connection
              </Button>
              <Button type="primary" htmlType="submit">
                Create Connection
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Connection Modal */}
      <Modal
        title="Edit Database Connection"
        open={showEditModal}
        onCancel={() => {
          setShowEditModal(false);
          setEditingConnection(null);
          connectionForm.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={connectionForm}
          layout="vertical"
          onFinish={handleUpdateConnection}
        >
          <Form.Item
            name="name"
            label="Connection Name"
            rules={[{ required: true, message: 'Please enter connection name' }]}
          >
            <Input placeholder="My Database" />
          </Form.Item>
          
          <Form.Item
            name="database_type"
            label="Database Type"
            rules={[{ required: true, message: 'Please select database type' }]}
          >
            <Select 
              placeholder="Select database type"
              onChange={(value) => handleDatabaseTypeChange(value, connectionForm)}
            >
              <Option value="postgresql">PostgreSQL (Default: 5432)</Option>
              <Option value="mysql">MySQL (Default: 3306)</Option>
              <Option value="mariadb">MariaDB (Default: 3306)</Option>
              <Option value="sqlite">SQLite (No port needed)</Option>
              <Option value="oracle">Oracle (Default: 1521)</Option>
              <Option value="mssql">Microsoft SQL Server (Default: 1433)</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="host"
            label="Host"
            rules={[{ required: true, message: 'Please enter host' }]}
          >
            <Input placeholder="localhost" />
          </Form.Item>

          <Form.Item
            noStyle
            shouldUpdate={(prevValues, currentValues) => 
              prevValues.database_type !== currentValues.database_type
            }
          >
            {({ getFieldValue }) => {
              const dbType = getFieldValue('database_type');
              const defaultPort = defaultPorts[dbType];
              
              if (dbType === 'sqlite') {
                return (
                  <Form.Item name="port" label="Port">
                    <Alert
                      message="SQLite doesn't require a port"
                      type="info"
                      showIcon
                      style={{ marginBottom: 0 }}
                    />
                  </Form.Item>
                );
              }
              
              const placeholder = defaultPort 
                ? `Default: ${defaultPort}` 
                : 'Enter port number';
              
              return (
                <Form.Item name="port" label="Port">
                  <Input 
                    type="number" 
                    placeholder={placeholder}
                  />
                </Form.Item>
              );
            }}
          </Form.Item>

          <Form.Item
            name="database_name"
            label="Database Name"
            rules={[{ required: true, message: 'Please enter database name' }]}
          >
            <Input placeholder="mydb" />
          </Form.Item>

          <Form.Item
            name="username"
            label="Username"
            rules={[{ required: true, message: 'Please enter username' }]}
          >
            <Input placeholder="user" />
          </Form.Item>

          <Form.Item
            name="password"
            label="Password (leave empty to keep current)"
          >
            <Input.Password placeholder="password" />
          </Form.Item>

          <Form.Item
            name="schema_name"
            label="Schema Name (optional)"
          >
            <Input placeholder="public" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button onClick={() => {
                setShowEditModal(false);
                setEditingConnection(null);
                connectionForm.resetFields();
              }}>
                Cancel
              </Button>
              <Button onClick={handleTestConnection} loading={testingConnection}>
                Test Connection
              </Button>
              <Button type="primary" htmlType="submit">
                Update Connection
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </Layout>
  );
};

export default DatabaseChat;

