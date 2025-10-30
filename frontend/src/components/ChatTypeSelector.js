import React from 'react';
import { Card, Row, Col, Typography, Space, Button } from 'antd';
import {
  FilePdfOutlined,
  FileExcelOutlined,
  FileTextOutlined,
  FileWordOutlined,
  FileOutlined,
  DatabaseOutlined,
  RocketOutlined
} from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const ChatTypeSelector = ({ onSelectChatType }) => {

  const chatTypes = [
    {
      id: 'pdf',
      title: 'Chat with PDF',
      icon: <FilePdfOutlined style={{ fontSize: '48px', color: '#ff4d4f' }} />,
      description: 'Upload PDF documents and chat with AI about their content. Perfect for research papers, reports, and documents.',
      color: '#ff4d4f',
      features: ['PDF Upload', 'Text Extraction', 'Vector Search', 'Context-Aware Chat']
    },
    {
      id: 'excel',
      title: 'Chat with Excel',
      icon: <FileExcelOutlined style={{ fontSize: '48px', color: '#52c41a' }} />,
      description: 'Upload Excel files and ask questions about your data. Analyze spreadsheets, tables, and datasets.',
      color: '#52c41a',
      features: ['Excel Upload', 'Data Analysis', 'Table Understanding', 'Query Data']
    },
    {
      id: 'text',
      title: 'Chat with Text File',
      icon: <FileTextOutlined style={{ fontSize: '48px', color: '#1890ff' }} />,
      description: 'Upload text files, documents, or paste content. Get insights and answers from your text documents.',
      color: '#1890ff',
      features: ['Text Upload', 'Content Analysis', 'Summarization', 'Q&A']
    },
    {
      id: 'word',
      title: 'Chat with Word Document',
      icon: <FileWordOutlined style={{ fontSize: '48px', color: '#2b579a' }} />,
      description: 'Upload Word documents (.docx) and have conversations about the content with AI assistance.',
      color: '#2b579a',
      features: ['Word Upload', 'Document Analysis', 'Content Extraction', 'Smart Chat']
    },
    {
      id: 'database',
      title: 'Chat with Database',
      icon: <DatabaseOutlined style={{ fontSize: '48px', color: '#722ed1' }} />,
      description: 'Connect to your database and query data using natural language. Get insights from your data.',
      color: '#722ed1',
      features: ['DB Connection', 'SQL Generation', 'Data Querying', 'Visualization']
    },
    {
      id: 'general',
      title: 'General Chat',
      icon: <FileOutlined style={{ fontSize: '48px', color: '#fa8c16' }} />,
      description: 'General conversation with AI without specific document context. Use for general questions and assistance.',
      color: '#fa8c16',
      features: ['Free Chat', 'General Q&A', 'AI Assistance', 'No Context']
    }
  ];

  const handleSelect = (chatType) => {
    if (onSelectChatType) {
      onSelectChatType(chatType);
    }
  };

  return (
    <div style={{ padding: '40px 24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <Title level={1} style={{ color: '#1890ff', marginBottom: '16px' }}>
            <RocketOutlined /> Choose Your Chat Type
          </Title>
          <Paragraph style={{ fontSize: '18px', color: '#666', maxWidth: '600px', margin: '0 auto' }}>
            Select the type of chat you want to start. Each type is optimized for different document formats and use cases.
          </Paragraph>
        </div>

        <Row gutter={[24, 24]}>
          {chatTypes.map((type) => (
            <Col xs={24} sm={12} lg={8} key={type.id}>
              <Card
                hoverable
                className="chat-type-card"
                style={{
                  height: '100%',
                  borderRadius: '12px',
                  border: `2px solid ${type.color}20`,
                  transition: 'all 0.3s ease'
                }}
                bodyStyle={{ padding: '24px' }}
                onClick={() => handleSelect(type.id)}
              >
                <Space direction="vertical" size="large" style={{ width: '100%', textAlign: 'center' }}>
                  <div>{type.icon}</div>
                  <div>
                    <Title level={4} style={{ marginBottom: '8px', color: type.color }}>
                      {type.title}
                    </Title>
                    <Paragraph style={{ color: '#666', marginBottom: '16px' }}>
                      {type.description}
                    </Paragraph>
                  </div>
                  <div style={{ textAlign: 'left' }}>
                    <Paragraph strong style={{ marginBottom: '8px', fontSize: '12px', color: '#8c8c8c' }}>
                      Features:
                    </Paragraph>
                    <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px', color: '#595959' }}>
                      {type.features.map((feature, idx) => (
                        <li key={idx}>{feature}</li>
                      ))}
                    </ul>
                  </div>
                  <Button
                    type="primary"
                    size="large"
                    style={{
                      backgroundColor: type.color,
                      borderColor: type.color,
                      width: '100%',
                      marginTop: '8px'
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelect(type.id);
                    }}
                  >
                    Start {type.title}
                  </Button>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </div>
    </div>
  );
};

export default ChatTypeSelector;

