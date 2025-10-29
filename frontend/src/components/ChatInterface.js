import React, { useState, useRef, useEffect } from 'react';
import {
  Layout,
  Menu,
  Input,
  Button,
  Avatar,
  Card,
  Space,
  Typography,
  Divider,
  List,
  Dropdown,
  Drawer,
  Tooltip,
  Select
} from 'antd';
import {
  MessageOutlined,
  RocketOutlined,
  SettingOutlined,
  MenuOutlined,
  SendOutlined,
  PlusOutlined,
  FileOutlined,
  ApiOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  UserOutlined,
  RobotOutlined,
  ClockCircleOutlined,
  DeleteOutlined
} from '@ant-design/icons';
import Markdown from 'react-markdown';
import { llmModelsAPI } from '../services/api';
import '../styles/ChatInterface.css';

const { Header, Content, Footer, Sider } = Layout;
const { TextArea } = Input;
const { Text } = Typography;
const { Option } = Select;

const ChatInterface = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedChat, setSelectedChat] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Chat history - user's previous chat sessions
  const [chatHistory, setChatHistory] = useState([
    { 
      id: 1, 
      title: 'Database Query Analysis', 
      lastMessage: 'How do I query the user table?', 
      timestamp: new Date(Date.now() - 86400000),
      messageCount: 12
    },
    { 
      id: 2, 
      title: 'AI Assistant Chat', 
      lastMessage: 'Explain RAG systems', 
      timestamp: new Date(Date.now() - 3600000),
      messageCount: 8
    },
    { 
      id: 3, 
      title: 'News Article Analysis', 
      lastMessage: 'Summarize the latest tech news', 
      timestamp: new Date(Date.now() - 7200000),
      messageCount: 15
    },
    { 
      id: 4, 
      title: 'API Documentation', 
      lastMessage: 'How to use the chat endpoint?', 
      timestamp: new Date(Date.now() - 10800000),
      messageCount: 5
    },
    { 
      id: 5, 
      title: 'Data Exploration', 
      lastMessage: 'Show me trending topics', 
      timestamp: new Date(Date.now() - 172800000),
      messageCount: 20
    }
  ]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoadingModels(true);
    try {
      const models = await llmModelsAPI.getAll();
      const activeModels = models.filter(model => model.is_active === 1);
      setAvailableModels(activeModels);
      if (activeModels.length > 0 && !selectedModel) {
        setSelectedModel(activeModels[0].id);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    } finally {
      setLoadingModels(false);
    }
  };

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    // Create new chat if none selected
    let currentChat = selectedChat;
    if (!currentChat) {
      const newChatId = Date.now();
      currentChat = {
        id: newChatId,
        title: inputValue.length > 30 ? inputValue.substring(0, 30) + '...' : inputValue,
        lastMessage: inputValue,
        timestamp: new Date(),
        messageCount: 0
      };
      setSelectedChat(currentChat);
      setChatHistory(prev => [currentChat, ...prev]);
    }

    const newMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    const userMessage = inputValue;
    setChatMessages(prev => [...prev, newMessage]);
    setInputValue('');
    
    // Update chat history with latest message
    setChatHistory(prev => prev.map(chat => 
      chat.id === currentChat.id 
        ? { ...chat, lastMessage: userMessage, timestamp: new Date(), messageCount: chat.messageCount + 1 }
        : chat
    ));
    
    // Simulate assistant response
    setTimeout(() => {
      const assistantResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `I understand you're asking about: "${userMessage}"\n\nLet me help you with that. This is a simulated response. The actual implementation will connect to your RAG system and LLM models.`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, assistantResponse]);
      
      // Update message count
      setChatHistory(prev => prev.map(chat => 
        chat.id === currentChat.id 
          ? { ...chat, messageCount: chat.messageCount + 2 }
          : chat
      ));
    }, 1000);

    inputRef.current?.focus();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleChatClick = (chat) => {
    setSelectedChat(chat);
    setChatMessages([]); // In real implementation, load messages for this chat
  };

  const handleNewChat = () => {
    setSelectedChat(null);
    setChatMessages([]);
    inputRef.current?.focus();
  };

  const handleDeleteChat = (chatId, e) => {
    e.stopPropagation();
    setChatHistory(prev => prev.filter(chat => chat.id !== chatId));
    if (selectedChat?.id === chatId) {
      setSelectedChat(null);
      setChatMessages([]);
    }
  };

  const formatTimestamp = (date) => {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  const sidebarMenuItems = [
    {
      key: '1',
      icon: <MessageOutlined />,
      label: 'Chat',
    },
    {
      key: '2',
      icon: <RocketOutlined />,
      label: 'Explore',
    },
    {
      key: '3',
      icon: <PlusOutlined />,
      label: 'Construct App',
    },
  ];

  return (
    <Layout className="chat-interface-layout">
      {/* Left Sidebar - Navigation */}
      <Sider
        className="left-sidebar"
        width={200}
        collapsed={sidebarCollapsed}
        collapsible={false}
        trigger={null}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        <div className="logo-container">
          {!sidebarCollapsed && (
            <div className="logo-content">
              <DatabaseOutlined className="logo-icon" />
              <Text className="logo-text">RAG Chat</Text>
            </div>
          )}
          {sidebarCollapsed && (
            <DatabaseOutlined className="logo-icon-collapsed" />
          )}
        </div>

        <Menu
          theme="light"
          mode="inline"
          selectedKeys={['1']}
          className="nav-menu"
          items={sidebarMenuItems}
        />

        <div className="sidebar-footer">
          <Tooltip title="Settings" placement="right">
            <Button
              type="text"
              icon={<SettingOutlined />}
              size="large"
              className="settings-btn"
            />
          </Tooltip>
        </div>
      </Sider>

      {/* Middle Panel - Chat History */}
      <Layout
        className="middle-panel-layout"
        style={{ marginLeft: sidebarCollapsed ? 80 : 200 }}
      >
        <div className="middle-panel">
          <div className="panel-header">
            <Button
              type="text"
              icon={<MenuOutlined />}
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="toggle-btn"
            />
            <Text strong className="panel-title">Chat History</Text>
            <Button
              type="text"
              icon={<PlusOutlined />}
              onClick={handleNewChat}
              className="new-chat-btn"
              title="New Chat"
            />
          </div>

          <div className="dialog-list-container">
            <List
              className="dialog-list"
              dataSource={chatHistory}
              renderItem={(item) => (
                <List.Item
                  className={`dialog-item ${selectedChat?.id === item.id ? 'active' : ''}`}
                  onClick={() => handleChatClick(item)}
                >
                  <div className="chat-history-item">
                    <div className="chat-history-header">
                      <Space className="chat-title-space">
                        <MessageOutlined className="chat-icon" />
                        <Text className="chat-title" ellipsis={{ tooltip: item.title }}>
                          {item.title}
                        </Text>
                      </Space>
                      <Button
                        type="text"
                        icon={<DeleteOutlined />}
                        size="small"
                        className="delete-chat-btn"
                        onClick={(e) => handleDeleteChat(item.id, e)}
                        title="Delete chat"
                      />
                    </div>
                    <div className="chat-history-preview">
                      <Text className="chat-preview" ellipsis>
                        {item.lastMessage}
                      </Text>
                    </div>
                    <div className="chat-history-footer">
                      <Space size="small">
                        <ClockCircleOutlined className="time-icon" />
                        <Text className="chat-time">{formatTimestamp(item.timestamp)}</Text>
                        {item.messageCount > 0 && (
                          <Text className="chat-count">{item.messageCount} messages</Text>
                        )}
                      </Space>
                    </div>
                  </div>
                </List.Item>
              )}
            />
          </div>
        </div>
      </Layout>

      {/* Right Panel - Main Chat Area */}
      <Content
        className="main-chat-content"
        style={{ marginLeft: sidebarCollapsed ? 260 : 380 }}
      >
        <div className="chat-container">
          {/* Chat Header */}
          <Header className="chat-header">
            <Space>
              {selectedChat ? (
                <>
                  <MessageOutlined className="chat-icon" />
                  <Text strong>{selectedChat.title}</Text>
                </>
              ) : (
                <Text strong>Welcome to RAG Chat</Text>
              )}
            </Space>
            <Space>
              <Select
                value={selectedModel}
                onChange={setSelectedModel}
                loading={loadingModels}
                style={{ width: 200 }}
                placeholder="Select Model"
                className="model-selector"
              >
                {availableModels.map((model) => (
                  <Option key={model.id} value={model.id}>
                    <Space>
                      <ApiOutlined />
                      <span>{model.model_name}</span>
                      <Text type="secondary" style={{ fontSize: '10px' }}>
                        ({model.provider})
                      </Text>
                    </Space>
                  </Option>
                ))}
              </Select>
              {selectedChat && (
                <Text type="secondary" style={{ fontSize: '12px', marginLeft: 8 }}>
                  {selectedChat.messageCount || 0} messages
                </Text>
              )}
            </Space>
          </Header>

          {/* Messages Area */}
          <div className="messages-container">
            {chatMessages.length === 0 ? (
              <div className="empty-state">
                <RobotOutlined className="empty-icon" />
                <Text className="empty-text">
                  Start a conversation by typing your message below
                </Text>
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
                        <Avatar icon={<UserOutlined />} className="user-avatar" />
                      ) : (
                        <Avatar icon={<RobotOutlined />} className="assistant-avatar" />
                      )}
                    </div>
                    <div className="message-content">
                      <div className="message-bubble">
                        <Markdown>{message.content}</Markdown>
                      </div>
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

          {/* Input Area */}
          <Footer className="chat-input-footer">
            <div className="input-container">
              <TextArea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about your documents..."
                autoSize={{ minRows: 1, maxRows: 4 }}
                className="chat-input"
              />
              <Space className="input-actions">
                <Tooltip title="Upload File">
                  <Button
                    type="text"
                    icon={<FileOutlined />}
                    className="action-btn"
                  />
                </Tooltip>
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim()}
                  className="send-btn"
                >
                  Send
                </Button>
              </Space>
            </div>
          </Footer>
        </div>
      </Content>
    </Layout>
  );
};

export default ChatInterface;

