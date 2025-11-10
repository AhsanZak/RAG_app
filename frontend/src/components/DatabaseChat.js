import React, { useState, useRef, useEffect, useMemo } from 'react';
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
  Upload,
  Spin,
  Tooltip,
  Collapse
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
  SyncOutlined,
  FileTextOutlined,
  DownloadOutlined,
  EditOutlined,
  UploadOutlined
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
  const [schemaModalVisible, setSchemaModalVisible] = useState(false);
  const [schemaJsonContent, setSchemaJsonContent] = useState('');
  const [schemaTextContent, setSchemaTextContent] = useState('');
  const [schemaModalSaving, setSchemaModalSaving] = useState(false);
  const [schemaModalError, setSchemaModalError] = useState(null);
  const [schemaModalTab, setSchemaModalTab] = useState('structured');
  const [knowledgeItems, setKnowledgeItems] = useState([]);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [knowledgeTitle, setKnowledgeTitle] = useState('');
  const [knowledgeDescription, setKnowledgeDescription] = useState('');
  const [knowledgeFileList, setKnowledgeFileList] = useState([]);
  const [knowledgeUploading, setKnowledgeUploading] = useState(false);
  const [isProcessingMessage, setIsProcessingMessage] = useState(false);
  const [creatingSession, setCreatingSession] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const { Dragger } = Upload;

  const formatRelativeTime = useMemo(() => {
    const divisions = [
      { amount: 60, name: 'seconds' },
      { amount: 60, name: 'minutes' },
      { amount: 24, name: 'hours' },
      { amount: 7, name: 'days' },
      { amount: 4.34524, name: 'weeks' },
      { amount: 12, name: 'months' },
      { amount: Number.POSITIVE_INFINITY, name: 'years' },
    ];

    const rtf = typeof Intl !== 'undefined' && Intl.RelativeTimeFormat
      ? new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
      : null;

    return (value) => {
      if (!value) return 'Unknown time';
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return typeof value === 'string' ? value : 'Unknown time';
      }

      const now = new Date();
      let duration = (date.getTime() - now.getTime()) / 1000;

      for (const division of divisions) {
        if (Math.abs(duration) < division.amount) {
          if (rtf) {
            return rtf.format(Math.round(duration), division.name);
          }
          // Fallback text if Intl.RelativeTimeFormat unavailable
          const rounded = Math.round(Math.abs(duration));
          const unit = division.name.replace(/s$/, '');
          return duration <= 0
            ? `${rounded} ${unit}${rounded !== 1 ? 's' : ''} ago`
            : `in ${rounded} ${unit}${rounded !== 1 ? 's' : ''}`;
        }
        duration /= division.amount;
      }

      return date.toLocaleString();
    };
  }, []);

  const selectedModelLabel = useMemo(() => {
    if (!selectedModel) return null;
    const found = availableModels.find((model) => model.id === selectedModel);
    if (!found) return null;
    return found.display_name || found.name || found.model_name || found.id;
  }, [availableModels, selectedModel]);

  // Default ports for different database types
  const defaultPorts = {
    postgresql: 5432,
    mysql: 3306,
    mariadb: 3306,
    sqlite: null, // SQLite doesn't use a port
    oracle: 1521,
    mssql: 1433
  };

  const buildProcessTrace = ({
    metadata = {},
    response = {},
    sql,
    sqlExecuted,
    queryResultCount
  } = {}) => {
    const intent = metadata.intent || {};
    const intentType = intent.intent_type || (metadata.requires_sql ? 'sql_query' : 'general_chat');
    const requiresSql = response.requires_sql ?? metadata.requires_sql ?? intent.requires_sql ?? false;
    const tableSelectionData =
      response.table_selection ||
      metadata.table_selection ||
      {};
    const sqlValidation = metadata.sql_validation || {};

    const formatConfidence = (value) => {
      if (typeof value !== 'number' || Number.isNaN(value)) return undefined;
      return Math.round(value * 100);
    };

    const selectedTables = Array.isArray(tableSelectionData.selected_tables)
      ? tableSelectionData.selected_tables
      : [];

    return {
      intentType,
      intentConfidence: formatConfidence(intent.confidence),
      requiresSql: Boolean(requiresSql),
      semanticSearchUsed: metadata.used_semantic_search ?? false,
      relevantSchemaChunks: metadata.relevant_schema_parts_count ?? null,
      tableSelection: {
        enabled: selectedTables.length > 0,
        selectedTables,
        confidence: formatConfidence(tableSelectionData.confidence),
        method: tableSelectionData.method || tableSelectionData.selection_method || null
      },
      sqlGenerated: Boolean(sql),
      sqlExecuted: Boolean(sqlExecuted),
      queryResultCount: typeof queryResultCount === 'number' ? queryResultCount : undefined,
      validatedTables: sqlValidation.validated_tables || [],
      validatedColumns: sqlValidation.validated_columns || [],
      warnings: sqlValidation.warnings || [],
      errors: sqlValidation.errors || []
    };
  };

  const renderProcessTrace = (trace) => {
    if (!trace) return null;

    const printConfidence = (value) => (typeof value === 'number' ? `${value}%` : null);

    return (
      <Collapse size="small" ghost style={{ marginTop: 12 }}>
        <Collapse.Panel header="Process Trace" key="trace-panel">
          <Space direction="vertical" size={6} style={{ width: '100%' }}>
            <div>
              <Text strong>Intent:</Text>{' '}
              <Tag color={trace.requiresSql ? 'volcano' : 'blue'}>
                {trace.intentType || 'unknown'}
              </Tag>
              {typeof trace.intentConfidence === 'number' && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  Confidence {trace.intentConfidence}%
                </Text>
              )}
            </div>

            <div>
              <Text strong>SQL Required:</Text>{' '}
              <Tag color={trace.requiresSql ? 'red' : 'green'}>
                {trace.requiresSql ? 'Yes' : 'No'}
              </Tag>
            </div>

            <div>
              <Text strong>Semantic Search:</Text>{' '}
              <Tag color={trace.semanticSearchUsed ? 'purple' : 'default'}>
                {trace.semanticSearchUsed ? 'Used enriched schema vectors' : 'Not used'}
              </Tag>
              {typeof trace.relevantSchemaChunks === 'number' && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  {trace.relevantSchemaChunks} chunks
                </Text>
              )}
            </div>

            <div>
              <Text strong>Table Selection:</Text>{' '}
              <Tag color={trace.tableSelection?.enabled ? 'geekblue' : 'default'}>
                {trace.tableSelection?.enabled ? 'Enabled' : 'Not triggered'}
              </Tag>
              {trace.tableSelection?.method && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  {trace.tableSelection.method}
                </Text>
              )}
              {trace.tableSelection?.confidence !== undefined && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  Confidence {printConfidence(trace.tableSelection.confidence)}
                </Text>
              )}
              {trace.tableSelection?.selectedTables?.length > 0 && (
                <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
                  {trace.tableSelection.selectedTables.map((table) => (
                    <Tag key={table} color="blue">
                      {table}
                    </Tag>
                  ))}
                </Space>
              )}
            </div>

            <div>
              <Text strong>SQL Generation:</Text>{' '}
              <Tag color={trace.sqlGenerated ? 'processing' : 'default'}>
                {trace.sqlGenerated ? 'Generated' : 'Not needed'}
              </Tag>
              <Text strong style={{ marginLeft: 12 }}>Execution:</Text>{' '}
              <Tag color={trace.sqlExecuted ? 'processing' : 'default'}>
                {trace.sqlExecuted ? 'Executed' : 'Skipped'}
              </Tag>
              {typeof trace.queryResultCount === 'number' && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  {trace.queryResultCount} rows
                </Text>
              )}
            </div>

            {trace.validatedTables?.length > 0 && (
              <div>
                <Text strong>Validated Tables:</Text>
                <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
                  {trace.validatedTables.map((table) => (
                    <Tag key={table} color="cyan">
                      {table}
                    </Tag>
                  ))}
                </Space>
              </div>
            )}

            {trace.validatedColumns?.length > 0 && (
              <div>
                <Text strong>Validated Columns:</Text>
                <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
                  {trace.validatedColumns.map((column) => (
                    <Tag key={column} color="gold">
                      {column}
                    </Tag>
                  ))}
                </Space>
              </div>
            )}

            {trace.warnings?.length > 0 && (
              <div>
                <Text strong>Warnings:</Text>
                <Space direction="vertical" size={2} style={{ marginTop: 4 }}>
                  {trace.warnings.map((warn, index) => (
                    <Text type="warning" key={`warn-${index}`}>
                      • {warn}
                    </Text>
                  ))}
                </Space>
              </div>
            )}

            {trace.errors?.length > 0 && (
              <div>
                <Text strong>Errors:</Text>
                <Space direction="vertical" size={2} style={{ marginTop: 4 }}>
                  {trace.errors.map((err, index) => (
                    <Text type="danger" key={`err-${index}`}>
                      • {err}
                    </Text>
                  ))}
                </Space>
              </div>
            )}
          </Space>
        </Collapse.Panel>
      </Collapse>
    );
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

  useEffect(() => {
    if (currentConnection && currentConnection.id) {
      loadKnowledge(currentConnection.id);
    } else {
      setKnowledgeItems([]);
      setKnowledgeTitle('');
      setKnowledgeDescription('');
      setKnowledgeFileList([]);
      setActiveTab('chat');
    }
  }, [currentConnection]);

  const loadModels = async () => {
    try {
      const models = await llmModelsAPI.getAll();
      const activeModels = models.filter(model => model.is_active === 1);
      setAvailableModels(activeModels);
      // Set default model if not already set
      if (activeModels.length > 0 && !selectedModel) {
        // Prefer Ollama models for database chat
        const ollamaModel = activeModels.find(m => m.provider === 'ollama');
        setSelectedModel(ollamaModel ? ollamaModel.id : activeModels[0].id);
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

  const handleOpenSchemaModal = () => {
    if (!schemaData) {
      message.warning('Schema not available yet. Please extract schema first.');
      return;
    }

    try {
      const formattedJson = JSON.stringify(schemaData.schema_data || {}, null, 2);
      setSchemaJsonContent(formattedJson);
      setSchemaTextContent(schemaData.schema_text || '');
      setSchemaModalError(null);
      setSchemaModalTab('structured');
      setSchemaModalVisible(true);
    } catch (error) {
      console.error('Failed to prepare schema JSON for editing:', error);
      message.error('Failed to prepare schema for editing');
    }
  };

  const handleCloseSchemaModal = () => {
    if (schemaModalSaving) {
      return;
    }
    setSchemaModalVisible(false);
    setSchemaModalError(null);
  };

  const handleDownloadSchema = (format = 'json') => {
    if (!schemaData) {
      message.warning('Schema not available to download yet.');
      return;
    }

    try {
      let content = '';
      let mimeType = 'application/json';
      let extension = 'json';

      if (format === 'json') {
        content = JSON.stringify(schemaData.schema_data || {}, null, 2);
      } else if (format === 'text') {
        content = schemaData.schema_text || 'No schema summary available.';
        mimeType = 'text/plain';
        extension = 'txt';
      } else {
        throw new Error('Unsupported download format');
      }

      const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const safeConnectionName = currentConnection?.name
        ? currentConnection.name.replace(/\s+/g, '_').toLowerCase()
        : 'database';
      link.download = `${safeConnectionName}_schema.${extension}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success(`Schema ${format === 'json' ? 'JSON' : 'summary'} downloaded`);
    } catch (error) {
      console.error('Failed to download schema:', error);
      message.error('Failed to download schema');
    }
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

  const loadSessions = async (connectionId = null) => {
    try {
      const sessionsData = await databaseChatAPI.getSessions(1, connectionId);
      // Filter for database sessions only
      const dbSessions = sessionsData.filter(s => s.session_type === 'database');
      setSessions(dbSessions);
      if (connectionId) {
        loadKnowledge(connectionId);
      }
      
      // If there's a current session and it's in the list, keep it selected
      // Otherwise, select the most recent session
      if (dbSessions.length > 0) {
        const existingSession = dbSessions.find(s => s.id === currentSession?.id);
        if (existingSession) {
          setCurrentSession(existingSession);
          await handleSelectSession(existingSession);
        } else {
          // Select the most recent session
          const mostRecent = dbSessions.sort((a, b) => 
            new Date(b.created_at) - new Date(a.created_at)
          )[0];
          setCurrentSession(mostRecent);
          await handleSelectSession(mostRecent);
        }
      } else {
        setCurrentSession(null);
        setChatMessages([]);
        setSessionReady(false);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
      setSessions([]);
      setCurrentSession(null);
    }
  };

  const handleCreateNewSession = async () => {
    if (!currentConnection) {
      message.warning('Select a connection first');
      return;
    }
    if (!schemaData || !schemaData.schema_data) {
      message.warning('Extract and process the schema before creating a new session');
      return;
    }
    if (!selectedModel) {
      message.warning('Select an LLM model first');
      return;
    }

    setCreatingSession(true);
    setChatMessages([]);
    setSessionReady(false);

    try {
      const timestampLabel = new Date().toLocaleString();
      const sessionName = `${currentConnection.name} - Session ${timestampLabel}`;

      const result = await databaseChatAPI.createSession({
        connectionId: currentConnection.id,
        sessionName,
        userId: 1,
        llmModelId: selectedModel,
        embeddingModelName: selectedEmbeddingModelName || 'all-MiniLM-L6-v2',
      });

      message.success('New session created');

      const sessionsData = await databaseChatAPI.getSessions(1, currentConnection.id);
      const dbSessions = sessionsData.filter(s => s.session_type === 'database');
      setSessions(dbSessions);

      await loadKnowledge(currentConnection.id);

      const newSessionId = result.session?.id || result.session_id;
      let sessionToSelect = dbSessions.find(s => s.id === newSessionId);

      if (!sessionToSelect && dbSessions.length > 0) {
        sessionToSelect = dbSessions
          .slice()
          .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))[0];
      }

      if (sessionToSelect) {
        setCurrentSession(sessionToSelect);
        setChatMessages([]);
        await handleSelectSession(sessionToSelect);
      } else {
        setCurrentSession(null);
        setChatMessages([]);
        setSessionReady(false);
      }
    } catch (error) {
      console.error('Failed to create new session:', error);
      message.error(error.response?.data?.detail || 'Failed to create new session');
    } finally {
      setCreatingSession(false);
    }
  };

  const loadKnowledge = async (connectionId) => {
    if (!connectionId) return;
    setKnowledgeLoading(true);
    try {
      const data = await databaseChatAPI.getKnowledge(connectionId);
      setKnowledgeItems(data || []);
    } catch (error) {
      console.error('Error loading knowledge documents:', error);
      message.error('Failed to load knowledge documents');
      setKnowledgeItems([]);
    } finally {
      setKnowledgeLoading(false);
    }
  };

  const handleConfirmKnowledgeUpload = async () => {
    if (!currentConnection) {
      message.warning('Select a database connection first');
      return;
    }
    if (!knowledgeFileList.length) {
      message.info('Choose a knowledge document to upload');
      return;
    }

    const uploadFile = knowledgeFileList[0].originFileObj || knowledgeFileList[0];
    setKnowledgeUploading(true);
    try {
      await databaseChatAPI.uploadKnowledge({
        connectionId: currentConnection.id,
        sessionId: currentSession?.id,
        title: knowledgeTitle,
        description: knowledgeDescription,
        file: uploadFile,
      });
      message.success('Knowledge document uploaded');
      setKnowledgeTitle('');
      setKnowledgeDescription('');
      setKnowledgeFileList([]);
      await loadKnowledge(currentConnection.id);
    } catch (error) {
      console.error('Failed to upload knowledge document:', error);
      message.error(error.response?.data?.detail || 'Failed to upload knowledge document');
    } finally {
      setKnowledgeUploading(false);
    }
  };

  const handleDeleteKnowledge = async (item) => {
    if (!currentConnection) return;
    try {
      await databaseChatAPI.deleteKnowledge(item.id);
      message.success('Knowledge document deleted');
      await loadKnowledge(currentConnection.id);
    } catch (error) {
      console.error('Failed to delete knowledge document:', error);
      message.error('Failed to delete knowledge document');
    }
  };

  const handleDownloadKnowledge = (item) => {
    const url = databaseChatAPI.getKnowledgeDownloadUrl(item.id);
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const handleSelectSession = async (session) => {
    setCurrentSession(session);
    setSessionReady(session.hasVectorizedData || false);
    
    try {
      // Load messages for this session
      const messagesData = await databaseChatAPI.getSessionMessages(session.id);
      const formattedMessages = messagesData.map((msg) => {
        const metadata = msg.metadata || {};
        const queryMeta = metadata.query_result || {};
        const queryData = Array.isArray(queryMeta.data) ? queryMeta.data : [];
        const queryColumns =
          Array.isArray(queryMeta.columns) && queryMeta.columns.length > 0
            ? queryMeta.columns
            : (queryData.length > 0 && typeof queryData[0] === 'object'
                ? Object.keys(queryData[0])
                : []);
        const rowCount = typeof queryMeta.row_count === 'number'
          ? queryMeta.row_count
          : queryData.length;
        const sql = metadata.sql;
        const sqlExecuted = Boolean(sql) && (rowCount > 0 || queryData.length > 0);
        const processTrace = msg.role === 'assistant'
          ? buildProcessTrace({
              metadata,
              response: {
                table_selection: metadata.table_selection,
                requires_sql: metadata.requires_sql
              },
              sql,
              sqlExecuted,
              queryResultCount: rowCount
            })
          : null;

        return {
          id: msg.id,
          role: msg.role,
          content: msg.message || msg.content,
          timestamp: new Date(msg.created_at),
          metadata,
          sql,
          sqlExecuted,
          queryResult: queryData,
          queryResultCount: rowCount,
          queryResultColumns: queryColumns,
          processTrace,
        };
      });
      setChatMessages(formattedMessages);
    } catch (error) {
      console.error('Error loading session messages:', error);
      setChatMessages([]);
    }
  };

  const handleSelectConnection = async (connection) => {
    setCurrentConnection(connection);
    setChatMessages([]);
    setSessionReady(false);
    setCurrentSession(null);
    setKnowledgeItems([]);
    setKnowledgeTitle('');
    setKnowledgeDescription('');
    setActiveTab('chat');
    
    // Try to load existing schema
    try {
      const schema = await databaseChatAPI.getSchema(connection.id);
      if (schema && schema.schema_data) {
        setSchemaData(schema);
        // Check if schema has a session_id (processed schema)
        if (schema.session_id) {
          // Load sessions for this connection
          await loadSessions(connection.id);
        }
      } else {
        setSchemaData(null);
        setSessions([]);
      }
    } catch (error) {
      // No schema found, that's okay
      setSchemaData(null);
      setSessions([]);
    }
    
    await loadKnowledge(connection.id);
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

  const handleProcessSchema = async (schemaOverride = null) => {
    const activeSchema = schemaOverride || schemaData;

    if (!activeSchema || !currentConnection) {
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
        // Reload sessions to get the new session with all details
        const sessionsData = await databaseChatAPI.getSessions(1, currentConnection.id);
        const dbSessions = sessionsData.filter(s => s.session_type === 'database');
        setSessions(dbSessions);
        
        // Find the newly created session
        const newSession = dbSessions.find(s => s.id === result.session_id) || {
          id: result.session_id,
          name: sessionName,
          createdAt: new Date(),
          status: 'ready',
          hasVectorizedData: true,
          session_type: 'database',
          connection_id: currentConnection.id,
          message_count: 0
        };
        
        setCurrentSession(newSession);
        setSessionReady(true);
        
        // Load messages for the new session (will be empty initially)
        await handleSelectSession(newSession);
        
        // Inject preview as first assistant message if available
        if (result.preview && result.preview.summary) {
          const assistantPreview = {
            id: `preview_${Date.now()}`,
            role: 'assistant',
            content: result.preview.summary,
            timestamp: new Date()
          };
          setChatMessages(prev => [assistantPreview, ...prev]);
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

  const handleSaveSchemaEdits = async ({ vectorizeAfterSave = false } = {}) => {
    if (!currentConnection) {
      message.warning('Select a connection before editing schema.');
      return;
    }

    try {
      setSchemaModalError(null);
      let parsedSchema = null;

      try {
        parsedSchema = JSON.parse(schemaJsonContent || '{}');
      } catch (parseError) {
        setSchemaModalError('Schema JSON is invalid. Please fix the JSON before saving.');
        return;
      }

      setSchemaModalSaving(true);
      const payload = {
        connection_id: currentConnection.id,
        schema_data: parsedSchema,
        schema_text: schemaTextContent
      };

      const saveResult = await databaseChatAPI.saveSchema(payload);
      if (saveResult.success) {
        const updatedSchema = {
          ...(schemaData || {}),
          schema_data: parsedSchema,
          schema_text: schemaTextContent
        };
        setSchemaData(updatedSchema);
        setSessionReady(false);
        setCurrentSession(null);
        setChatMessages([]);

        message.success('Schema updated. Please vectorize to refresh the chat context.');

        if (vectorizeAfterSave) {
          setSchemaModalVisible(false);
          await handleProcessSchema(updatedSchema);
        }
      }
    } catch (error) {
      console.error('Failed to save schema edits:', error);
      const detail = error.response?.data?.detail;
      setSchemaModalError(detail || 'Failed to save schema changes. Please try again.');
    } finally {
      setSchemaModalSaving(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !currentSession) return;
    
    if (!sessionReady) {
      message.warning('Please process schema first before chatting');
      return;
    }

    if (!currentSession.id) {
      message.warning('Session not ready. Please process schema first.');
      return;
    }

    // Use selected model or default to first available model
    const modelToUse = selectedModel || (availableModels.length > 0 ? availableModels[0].id : null);
    if (!modelToUse) {
      message.warning('Please select an LLM model first');
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

    setIsProcessingMessage(true);
    try {
      const response = await databaseChatAPI.chat(
        currentSession.id,
        messageText,
        modelToUse
      );

      // Debug: Log the response to see what we're getting
      console.log('=== Database Chat Response Debug ===');
      console.log('Full response object:', response);
      console.log('query_result:', response.query_result);
      console.log('query_result type:', typeof response.query_result);
      console.log('query_result is array:', Array.isArray(response.query_result));
      console.log('query_result length:', response.query_result?.length || 0);
      console.log('query_result_count:', response.query_result_count);
      console.log('query_result_columns:', response.query_result_columns);
      console.log('sql_executed:', response.sql_executed);
      console.log('sql:', response.sql);
      console.log('===================================');

      // Ensure query_result is always an array
      let queryResultArray = [];
      if (response.query_result) {
        if (Array.isArray(response.query_result)) {
          queryResultArray = response.query_result;
        } else if (typeof response.query_result === 'object' && response.query_result.data) {
          // Handle case where data is nested
          queryResultArray = Array.isArray(response.query_result.data) ? response.query_result.data : [];
        }
      }

      // Extract columns
      let queryResultColumns = response.query_result_columns || [];
      if (queryResultColumns.length === 0 && queryResultArray.length > 0 && typeof queryResultArray[0] === 'object') {
        queryResultColumns = Object.keys(queryResultArray[0]);
      }

      const processTrace = buildProcessTrace({
        metadata: response.metadata || {},
        response,
        sql: response.sql,
        sqlExecuted: response.sql_executed || false,
        queryResultCount: response.query_result_count || queryResultArray.length
      });

      const assistantResponse = {
        id: response.message_id || Date.now() + 1,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        sources: response.sources || [],
        sql: response.sql,
        sqlExecuted: response.sql_executed || false,
        queryResult: queryResultArray,
        queryResultCount: response.query_result_count || queryResultArray.length,
        queryResultColumns: queryResultColumns,
        metadata: response.metadata || {},
        processTrace
      };

      console.log('=== Processed Assistant Response ===');
      console.log('queryResult length:', assistantResponse.queryResult.length);
      console.log('queryResultCount:', assistantResponse.queryResultCount);
      console.log('queryResultColumns:', assistantResponse.queryResultColumns);
      console.log('sqlExecuted:', assistantResponse.sqlExecuted);
      console.log('===================================');
      
      setChatMessages(prev => [...prev, assistantResponse]);
      setIsProcessingMessage(false);
      
      // Reload sessions to update message count
      if (currentConnection) {
        await loadSessions(currentConnection.id);
      }
    } catch (error) {
      console.error('Chat error:', error);
      message.error(error.response?.data?.detail || 'Failed to send message');
      
      // Remove user message on error
      setChatMessages(prev => prev.filter(m => m.id !== userMessage.id));
      setIsProcessingMessage(false);
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

  const knowledgeUploadProps = {
    accept: '.pdf,.json,.txt,.md,.csv,.yaml,.yml,.sql',
    multiple: false,
    maxCount: 1,
    fileList: knowledgeFileList,
    beforeUpload: (file) => {
      if (!currentConnection) {
        message.warning('Select a database connection first');
        return Upload.LIST_IGNORE;
      }
      const maxBytes = 10 * 1024 * 1024;
      if (file.size && file.size > maxBytes) {
        message.error('File size must be 10MB or less');
        return Upload.LIST_IGNORE;
      }
      setKnowledgeFileList([file]);
      return false; // prevent auto upload
    },
    onChange: (info) => {
      setKnowledgeFileList(info.fileList.slice(-1));
    },
    onRemove: () => {
      setKnowledgeFileList([]);
    },
  };

  const renderKnowledgeTab = () => {
    if (!currentConnection) {
      return (
        <Alert
          message="Select a connection"
          description="Choose a database connection to upload schema helpers or query examples."
          type="info"
          showIcon
        />
      );
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, height: '100%', minHeight: 0, overflow: 'hidden' }}>
        <Card
          title="Upload Knowledge Document"
          extra={
            <Tag color="purple">
              Embedding: {selectedEmbeddingModelName || 'Not selected'}
            </Tag>
          }
          bodyStyle={{ display: 'flex', flexDirection: 'column', gap: 16 }}
        >
          <Input
            placeholder="Title (optional)"
            value={knowledgeTitle}
            onChange={(e) => setKnowledgeTitle(e.target.value)}
          />
          <Input.TextArea
            rows={3}
            placeholder="Description or purpose (optional)"
            value={knowledgeDescription}
            onChange={(e) => setKnowledgeDescription(e.target.value)}
          />
          <Dragger {...knowledgeUploadProps} style={{ padding: 16 }}>
            <p className="ant-upload-drag-icon">
              <DatabaseOutlined style={{ color: '#722ed1' }} />
            </p>
            <p className="ant-upload-text">Click or drag files to this area</p>
            <p className="ant-upload-hint">
              Supported formats: PDF, JSON, TXT, Markdown, CSV, YAML, SQL (max 10MB)
            </p>
          </Dragger>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography.Text type="secondary">
              {knowledgeFileList.length > 0
                ? `Selected: ${knowledgeFileList[0].name}`
                : 'No file selected'}
            </Typography.Text>
            <Space>
              {knowledgeFileList.length > 0 && (
                <Button onClick={() => setKnowledgeFileList([])}>Clear</Button>
              )}
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={handleConfirmKnowledgeUpload}
                disabled={!knowledgeFileList.length}
                loading={knowledgeUploading}
              >
                Upload
              </Button>
            </Space>
          </div>
        </Card>

        <Card
          title="Knowledge Library"
          bodyStyle={{ padding: 0, display: 'flex', flexDirection: 'column', height: '100%' }}
          style={{ flex: 1, minHeight: 0 }}
        >
          <Spin spinning={knowledgeLoading} style={{ flex: 1, display: 'flex', minHeight: 0 }}>
            <List
              dataSource={knowledgeItems}
              locale={{ emptyText: 'No knowledge documents uploaded yet.' }}
              style={{ flex: 1, maxHeight: '100%', overflowY: 'auto' }}
              renderItem={(item) => (
                <List.Item
                  key={item.id}
                  style={{ paddingInline: 24 }}
                  actions={[
                    <Tooltip title="Download" key="download">
                      <Button
                        type="link"
                        icon={<DownloadOutlined />}
                        onClick={() => handleDownloadKnowledge(item)}
                      />
                    </Tooltip>,
                    <Tooltip title="Delete" key="delete">
                      <Button
                        type="link"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => handleDeleteKnowledge(item)}
                      />
                    </Tooltip>,
                  ]}
                >
                  <List.Item.Meta
                    title={
                      <Space wrap>
                        <FileTextOutlined />
                        <Typography.Text strong>
                          {item.title || item.original_filename || 'Untitled document'}
                        </Typography.Text>
                        {item.file_type && <Tag color="geekblue">{item.file_type.toUpperCase()}</Tag>}
                        {item.embedding_model && (
                          <Tag color="cyan">{item.embedding_model}</Tag>
                        )}
                      </Space>
                    }
                    description={
                      <Space direction="vertical" size="small">
                        <Typography.Text>
                          {item.description || 'No description provided.'}
                        </Typography.Text>
                        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                          {item.original_filename || 'Unknown file'} • {item.file_type?.toUpperCase() || 'FILE'} • Uploaded{' '}
                          {item.created_at_human || formatRelativeTime(item.created_at)}
                        </Typography.Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Spin>
        </Card>
      </div>
    );
  };

  const renderChatTab = () => {
    if (!currentConnection) {
      return (
        <div className="empty-chat-state">
          <DatabaseOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
          <Title level={3}>Select a Connection</Title>
          <Paragraph>
            Select a database connection from the left panel to start chatting with your database schema.
          </Paragraph>
        </div>
      );
    }

    if (!schemaData) {
      return (
        <div className="empty-chat-state">
          <SyncOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
          <Title level={3}>Extract Schema</Title>
          <Paragraph>
            Click "Extract Schema" in the left panel to extract and analyze your database schema.
          </Paragraph>
        </div>
      );
    }

    if (!sessionReady) {
      return (
        <div className="empty-chat-state">
          <RobotOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
          <Title level={3}>Process Schema</Title>
          <Paragraph>
            Click "Process & Vectorize Schema" in the left panel to enable chat functionality.
          </Paragraph>
        </div>
      );
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
        <Space size={[8, 8]} wrap style={{ marginBottom: 16 }}>
          <Tag color="purple">
            Embedding: {selectedEmbeddingModelName || 'Not selected'}
          </Tag>
          {selectedModelLabel && (
            <Tag color="geekblue">
              LLM: {selectedModelLabel}
            </Tag>
          )}
        </Space>

        <div
          className="chat-messages-container"
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            paddingRight: 8,
            marginBottom: 16,
          }}
        >
          {isProcessingMessage && (
            <Card
              size="small"
              bordered={false}
              style={{
                marginBottom: 12,
                background: '#f6ffed',
                border: '1px solid #b7eb8f'
              }}
            >
              <Space align="start">
                <Spin
                  indicator={<LoadingOutlined style={{ fontSize: 18, color: '#389e0d' }} spin />}
                />
                <div>
                  <Text strong>Working on your request…</Text>
                  <Paragraph style={{ marginBottom: 0, marginTop: 4 }}>
                    Detecting intent → Selecting relevant tables → Generating SQL → Executing query (if needed)
                  </Paragraph>
                </div>
              </Space>
            </Card>
          )}
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
                    {message.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                  </div>
                  <div className="message-content">
                    <div className="message-bubble">
                      <Markdown>{message.content}</Markdown>
                      
                      {message.role === 'assistant' && message.processTrace && renderProcessTrace(message.processTrace)}

                      {message.sql && (
                        <div style={{ marginTop: 12, padding: 8, background: '#f5f5f5', borderRadius: 4 }}>
                          <Text strong style={{ fontSize: '12px' }}>SQL Query:</Text>
                          <pre style={{ marginTop: 4, fontSize: '11px', whiteSpace: 'pre-wrap' }}>
                            {message.sql}
                          </pre>
                        </div>
                      )}
                      
                      {message.sqlExecuted && Array.isArray(message.queryResult) && message.queryResult.length > 0 && (
                        <div style={{ marginTop: 12 }}>
                          <Text strong style={{ fontSize: '12px' }}>
                            Query Results {message.queryResultCount && message.queryResultCount !== message.queryResult.length 
                              ? `(${message.queryResultCount} total, showing ${message.queryResult.length})` 
                              : `(${message.queryResult.length} ${message.queryResult.length === 1 ? 'row' : 'rows'})`}:
                          </Text>
                          <div style={{ marginTop: 8, maxHeight: '500px', overflow: 'auto', border: '1px solid #d9d9d9', borderRadius: 4, background: '#fff' }}>
                            <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                              <thead>
                                <tr style={{ background: '#f0f0f0', position: 'sticky', top: 0, zIndex: 1 }}>
                                  {(Array.isArray(message.queryResultColumns) && message.queryResultColumns.length > 0
                                    ? message.queryResultColumns
                                    : Object.keys(message.queryResult[0] || {})
                                  ).map((key) => (
                                    <th key={key} style={{ padding: '8px', textAlign: 'left', border: '1px solid #ddd', fontWeight: 'bold' }}>
                                      {key}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {message.queryResult.map((row, idx) => (
                                  <tr key={idx} style={{ background: idx % 2 === 0 ? '#fff' : '#fafafa' }}>
                                    {(Array.isArray(message.queryResultColumns) && message.queryResultColumns.length > 0
                                      ? message.queryResultColumns
                                      : Object.keys(message.queryResult[0] || {})
                                    ).map((col, colIdx) => (
                                      <td key={colIdx} style={{ padding: '6px 8px', border: '1px solid #ddd', wordBreak: 'break-word' }}>
                                        {String(row && row[col] !== null && row[col] !== undefined ? row[col] : '')}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                            {message.queryResultCount && message.queryResultCount > message.queryResult.length && (
                              <div style={{ padding: '8px', background: '#f9f9f9', borderTop: '1px solid #ddd', textAlign: 'center' }}>
                                <Text type="secondary" style={{ fontSize: '11px' }}>
                                  Showing {message.queryResult.length} of {message.queryResultCount} rows
                                </Text>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                    {message.metadata?.debug_messages && Array.isArray(message.metadata.debug_messages) && (
                      <div style={{ marginTop: 8, fontSize: '11px', color: '#999' }}>
                        {message.metadata.debug_messages.join(' | ')}
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

        <Footer
          className="chat-input-footer"
          style={{ padding: '12px 0', background: 'transparent' }}
        >
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
      </div>
    );
  };

  return (
    <Layout
      className="pdf-chat-layout"
      style={{ height: '100vh', overflow: 'hidden' }}
    >
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

      <Content
        className="pdf-chat-content"
        style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}
      >
        <div
          className="pdf-chat-container"
          style={{ flex: 1, display: 'flex', gap: 24, height: '100%', overflow: 'hidden' }}
        >
          {/* Left Panel - Connections & Schema */}
          <div
            className="pdf-chat-sidebar"
            style={{ width: 360, minWidth: 320, maxWidth: 420, overflowY: 'auto', paddingRight: 8 }}
          >
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
                      <Button
                        block
                        icon={<FileTextOutlined />}
                        style={{ marginTop: 8 }}
                        onClick={handleOpenSchemaModal}
                      >
                        View / Edit Schema
                      </Button>
                      <Button
                        block
                        icon={<DownloadOutlined />}
                        style={{ marginTop: 8 }}
                        onClick={() => handleDownloadSchema('json')}
                      >
                        Download Schema JSON
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

            {/* Sessions List - similar to PDF Chat */}
            {currentConnection && (
              <Card
                className="sidebar-sessions"
                size="small"
                title="Chat Sessions"
                style={{ marginTop: 12 }}
                extra={
                  <Button
                    type="link"
                    size="small"
                    icon={<PlusOutlined />}
                    onClick={handleCreateNewSession}
                    loading={creatingSession}
                  >
                    New
                  </Button>
                }
              >
                {sessions.length === 0 ? (
                  <Alert
                    message="No chat sessions"
                    description="Use the New button to start a fresh conversation for this database."
                    type="info"
                    showIcon
                  />
                ) : (
                  <List
                    size="small"
                    dataSource={sessions}
                    renderItem={(session) => (
                      <List.Item
                        className={`session-item ${currentSession?.id === session.id ? 'active' : ''}`}
                        onClick={() => handleSelectSession(session)}
                      >
                        <List.Item.Meta
                          avatar={<DatabaseOutlined />}
                          title={<span style={{ fontSize: 13 }}>{session.name}</span>}
                          description={
                            <span style={{ fontSize: 12 }}>
                              {session.message_count || 0} messages
                              {session.hasVectorizedData ? ' - Ready' : ' - Processing'}
                            </span>
                          }
                        />
                        {session.hasVectorizedData && (
                          <Tag color="green" style={{ marginInlineStart: 6 }}>Ready</Tag>
                        )}
                      </List.Item>
                    )}
                  />
                )}
              </Card>
            )}
          </div>

          {/* Right Panel - Chat Area */}
          <div className="pdf-chat-main" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
            <Tabs
              className="database-chat-tabs"
              activeKey={activeTab}
              onChange={setActiveTab}
              type="card"
              tabBarGutter={16}
              style={{ flex: 1, minHeight: 0 }}
            >
              <Tabs.TabPane tab="Chat" key="chat" forceRender>
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                  {renderChatTab()}
                </div>
              </Tabs.TabPane>
              <Tabs.TabPane tab="Chat Settings" key="settings" disabled={!currentConnection} forceRender>
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
                  {renderKnowledgeTab()}
                </div>
              </Tabs.TabPane>
            </Tabs>
          </div>
        </div>
      </Content>

      {/* Schema View / Edit Modal */}
      <Modal
        title={
          <Space>
            <EditOutlined />
            <span>View / Edit Extracted Schema</span>
          </Space>
        }
        open={schemaModalVisible}
        onCancel={handleCloseSchemaModal}
        width={900}
        footer={[
          <Button
            key="download-json"
            icon={<DownloadOutlined />}
            onClick={() => handleDownloadSchema('json')}
            disabled={!schemaData}
          >
            Download JSON
          </Button>,
          <Button
            key="download-text"
            icon={<DownloadOutlined />}
            onClick={() => handleDownloadSchema('text')}
            disabled={!schemaData}
          >
            Download Summary
          </Button>,
          <Button key="cancel" onClick={handleCloseSchemaModal}>
            Cancel
          </Button>,
          <Button
            key="save"
            type="primary"
            loading={schemaModalSaving}
            onClick={() => handleSaveSchemaEdits({ vectorizeAfterSave: false })}
          >
            Save Changes
          </Button>,
          <Button
            key="save-vectorize"
            type="primary"
            loading={schemaModalSaving || processingSchema}
            onClick={() => handleSaveSchemaEdits({ vectorizeAfterSave: true })}
          >
            Save & Vectorize
          </Button>
        ]}
      >
        {!schemaData ? (
          <Alert
            message="Schema not available"
            description="Extract schema before attempting to view or edit it."
            type="warning"
            showIcon
          />
        ) : (
          <>
            <Alert
              message={`Connection: ${currentConnection?.name || 'Unknown'}`}
              description={
                <div>
                  <div>
                    Tables: {schemaData.schema_data?.metadata?.total_tables ?? 0} — Columns:{' '}
                    {schemaData.schema_data?.metadata?.total_columns ?? 0}
                  </div>
                  {!sessionReady && (
                    <div style={{ marginTop: 4 }}>
                      Schema changes require re-vectorization before chatting.
                    </div>
                  )}
                </div>
              }
              type={sessionReady ? 'info' : 'warning'}
              showIcon
              style={{ marginBottom: 16 }}
            />
            {schemaModalError && (
              <Alert
                type="error"
                message={schemaModalError}
                showIcon
                closable
                onClose={() => setSchemaModalError(null)}
                style={{ marginBottom: 16 }}
              />
            )}
            <Tabs activeKey={schemaModalTab} onChange={setSchemaModalTab}>
              <TabPane tab="Structured JSON" key="structured">
                <Input.TextArea
                  value={schemaJsonContent}
                  onChange={(e) => setSchemaJsonContent(e.target.value)}
                  autoSize={{ minRows: 18, maxRows: 32 }}
                  spellCheck={false}
                />
              </TabPane>
              <TabPane tab="Schema Summary" key="summary">
                <Input.TextArea
                  value={schemaTextContent}
                  onChange={(e) => setSchemaTextContent(e.target.value)}
                  autoSize={{ minRows: 18, maxRows: 32 }}
                  placeholder="Human-readable schema summary"
                />
              </TabPane>
            </Tabs>
          </>
        )}
      </Modal>

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

