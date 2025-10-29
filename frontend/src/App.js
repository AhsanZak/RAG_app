import React, { useState } from 'react';
import { Layout, Typography, Card, Row, Col, Button, Space, Menu } from 'antd';
import { 
  MessageOutlined, 
  DatabaseOutlined, 
  ApiOutlined, 
  RocketOutlined,
  SettingOutlined,
  HomeOutlined
} from '@ant-design/icons';
import LLMModelManagement from './components/LLMModelManagement';

const { Header, Content, Footer, Sider } = Layout;
const { Title, Paragraph } = Typography;

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const menuItems = [
    {
      key: 'home',
      icon: <HomeOutlined />,
      label: 'Home',
    },
    {
      key: 'models',
      icon: <ApiOutlined />,
      label: 'LLM Models',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  const renderContent = () => {
    switch (currentPage) {
      case 'models':
        return <LLMModelManagement />;
      case 'settings':
        return (
          <Card>
            <Title level={3}>Settings</Title>
            <Paragraph>Settings panel coming soon...</Paragraph>
          </Card>
        );
      default:
        return renderHomePage();
    }
  };

  const renderHomePage = () => (
    <div>
      <Row justify="center" style={{ marginBottom: '50px' }}>
        <Col span={24} style={{ textAlign: 'center' }}>
          <Title level={1} style={{ color: '#1890ff', marginBottom: '20px' }}>
            Welcome to RAG Chat System
          </Title>
          <Paragraph style={{ fontSize: '18px', color: '#666', maxWidth: '600px', margin: '0 auto' }}>
            A powerful AI-powered chat application that uses Retrieval-Augmented Generation (RAG) 
            to analyze and chat with news articles and other data from our vector database.
          </Paragraph>
        </Col>
      </Row>

      <Row gutter={[24, 24]} justify="center">
        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '250px', textAlign: 'center' }}
            cover={<MessageOutlined style={{ fontSize: '48px', color: '#1890ff', marginTop: '30px' }} />}
          >
            <Card.Meta 
              title="Intelligent Chat"
              description="Chat with AI that has access to a comprehensive knowledge base of news and articles"
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '250px', textAlign: 'center' }}
            cover={<DatabaseOutlined style={{ fontSize: '48px', color: '#52c41a', marginTop: '30px' }} />}
          >
            <Card.Meta 
              title="Vector Database"
              description="Powered by advanced vector embeddings for semantic search and retrieval"
            />
          </Card>
        </Col>
        
        <Col xs={24} sm={12} md={8}>
          <Card 
            hoverable 
            style={{ height: '250px', textAlign: 'center' }}
            cover={<ApiOutlined style={{ fontSize: '48px', color: '#722ed1', marginTop: '30px' }} />}
          >
            <Card.Meta 
              title="FastAPI Backend"
              description="Robust and scalable backend built with FastAPI for high-performance API responses"
            />
          </Card>
        </Col>
      </Row>

      <Row justify="center" style={{ marginTop: '50px' }}>
        <Col span={24} style={{ textAlign: 'center' }}>
          <Card style={{ maxWidth: '800px', margin: '0 auto' }}>
            <Title level={3}>Available API Endpoints</Title>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div style={{ textAlign: 'left' }}>
                <Title level={4}>Backend Endpoints:</Title>
                <ul style={{ textAlign: 'left', paddingLeft: '20px' }}>
                  <li><strong>GET /</strong> - Root endpoint (health check)</li>
                  <li><strong>GET /health</strong> - Health check endpoint</li>
                  <li><strong>POST /chat</strong> - Main chat endpoint for RAG conversations</li>
                  <li><strong>GET /api/endpoints</strong> - List all available endpoints</li>
                  <li><strong>GET /api/llm-models</strong> - List all LLM models</li>
                  <li><strong>GET /api/users</strong> - List all users</li>
                </ul>
              </div>
              <Space>
                <Button 
                  type="primary" 
                  size="large" 
                  icon={<ApiOutlined />}
                  onClick={() => setCurrentPage('models')}
                >
                  Manage LLM Models
                </Button>
                <Button 
                  type="primary" 
                  size="large" 
                  icon={<RocketOutlined />}
                  disabled
                >
                  Start Chatting (Coming Soon)
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        background: '#fff', 
        padding: '0 24px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
          <MessageOutlined /> RAG Chat Application
        </Title>
        <Menu
          mode="horizontal"
          selectedKeys={[currentPage]}
          items={menuItems}
          onClick={({ key }) => setCurrentPage(key)}
          style={{ background: 'transparent', border: 'none' }}
        />
      </Header>
      
      <Content style={{ padding: '50px 24px', background: '#f0f2f5' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {renderContent()}
        </div>
      </Content>
      
      <Footer style={{ textAlign: 'center', background: '#fff' }}>
        RAG Chat Application ©2024 - Built with FastAPI & React
      </Footer>
    </Layout>
  );
}

export default App;
