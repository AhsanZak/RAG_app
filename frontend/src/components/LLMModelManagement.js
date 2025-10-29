import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  message,
  Popconfirm,
  Tooltip,
  Typography,
  Divider,
  Alert
} from 'antd';
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { llmModelsAPI } from '../services/api';

const { Title, Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;

const LLMModelManagement = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [editingModel, setEditingModel] = useState(null);
  const [form] = Form.useForm();
  const [testingConnection, setTestingConnection] = useState(false);
  const [testResult, setTestResult] = useState(null);

  // Load models on component mount
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    setLoading(true);
    try {
      const data = await llmModelsAPI.getAll();
      setModels(data);
    } catch (error) {
      message.error('Failed to load LLM models');
      console.error('Error loading models:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAddModel = () => {
    setEditingModel(null);
    form.resetFields();
    setTestResult(null);
    setModalVisible(true);
  };

  const handleEditModel = (model) => {
    setEditingModel(model);
    form.setFieldsValue({
      ...model,
      is_active: model.is_active === 1
    });
    setTestResult(null);
    setModalVisible(true);
  };

  const handleDeleteModel = async (id) => {
    try {
      await llmModelsAPI.delete(id);
      message.success('Model deleted successfully');
      loadModels();
    } catch (error) {
      message.error('Failed to delete model');
      console.error('Error deleting model:', error);
    }
  };

  const handleModalSubmit = async (values) => {
    try {
      const modelData = {
        ...values,
        is_active: values.is_active ? 1 : 0
      };

      if (editingModel) {
        await llmModelsAPI.update(editingModel.id, modelData);
        message.success('Model updated successfully');
      } else {
        await llmModelsAPI.create(modelData);
        message.success('Model created successfully');
      }

      setModalVisible(false);
      form.resetFields();
      setTestResult(null);
      loadModels();
    } catch (error) {
      message.error(`Failed to ${editingModel ? 'update' : 'create'} model`);
      console.error('Error saving model:', error);
    }
  };

  const handleTestConnection = async () => {
    try {
      const values = await form.validateFields(['provider', 'base_url', 'model_name', 'api_key']);
      setTestingConnection(true);
      setTestResult(null);
      
      const testData = {
        provider: values.provider,
        base_url: values.base_url,
        model_name: values.model_name,
        api_key: values.api_key || ''
      };
      
      const result = await llmModelsAPI.testConnection(testData);
      setTestResult(result);
      
      if (result.success) {
        message.success(result.message);
      } else {
        message.error(result.message);
      }
    } catch (error) {
      if (error.errorFields) {
        message.error('Please fill in all required fields before testing');
      } else {
        message.error('Failed to test connection');
        console.error('Error testing connection:', error);
      }
    } finally {
      setTestingConnection(false);
    }
  };

  const getProviderConfig = (provider) => {
    const configs = {
      openai: {
        baseUrl: 'https://api.openai.com/v1',
        apiKeyPlaceholder: 'sk-...',
        description: 'OpenAI API Key (starts with sk-)',
        helpText: 'Get your API key from https://platform.openai.com/api-keys'
      },
      ollama: {
        baseUrl: 'http://localhost:11434',
        apiKeyPlaceholder: 'Not required for Ollama',
        description: 'Ollama Server URL',
        helpText: 'Default Ollama port is 11434. Make sure Ollama is running locally.'
      },
      azure: {
        baseUrl: 'https://your-resource.openai.azure.com/',
        apiKeyPlaceholder: 'Azure API Key',
        description: 'Azure OpenAI API Key',
        helpText: 'Get your API key from Azure OpenAI Studio'
      },
      anthropic: {
        baseUrl: 'https://api.anthropic.com',
        apiKeyPlaceholder: 'sk-ant-...',
        description: 'Anthropic API Key (starts with sk-ant-)',
        helpText: 'Get your API key from https://console.anthropic.com/'
      },
      local: {
        baseUrl: 'http://localhost:8000',
        apiKeyPlaceholder: 'Local API Key (if required)',
        description: 'Local Server URL',
        helpText: 'URL of your local LLM server'
      }
    };
    return configs[provider] || configs.local;
  };

  const getDefaultModels = (provider) => {
    const defaults = {
      openai: [
        { name: 'gpt-4-turbo-preview', description: 'GPT-4 Turbo Preview - Most capable model' },
        { name: 'gpt-4', description: 'GPT-4 - High capability model' },
        { name: 'gpt-3.5-turbo', description: 'GPT-3.5 Turbo - Fast and efficient' },
        { name: 'gpt-3.5-turbo-16k', description: 'GPT-3.5 Turbo 16K - Extended context' }
      ],
      ollama: [
        { name: 'llama2', description: 'Meta LLaMA 2 - 7B parameter model' },
        { name: 'llama2:13b', description: 'Meta LLaMA 2 - 13B parameter model' },
        { name: 'mistral', description: 'Mistral 7B - Efficient model' },
        { name: 'mixtral', description: 'Mixtral 8x7B - Mixture of Experts' },
        { name: 'codellama', description: 'Code LLaMA - Code generation model' },
        { name: 'vicuna', description: 'Vicuna - Fine-tuned LLaMA' }
      ],
      azure: [
        { name: 'gpt-4', description: 'GPT-4 via Azure OpenAI' },
        { name: 'gpt-35-turbo', description: 'GPT-3.5 Turbo via Azure OpenAI' }
      ],
      anthropic: [
        { name: 'claude-3-opus-20240229', description: 'Claude 3 Opus - Most capable' },
        { name: 'claude-3-sonnet-20240229', description: 'Claude 3 Sonnet - Balanced' },
        { name: 'claude-3-haiku-20240307', description: 'Claude 3 Haiku - Fast and efficient' }
      ],
      local: [
        { name: 'local-model', description: 'Custom local model' }
      ]
    };
    return defaults[provider] || defaults.local;
  };

  const columns = [
    {
      title: 'Model Name',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (text, record) => (
        <Space>
          <Text strong>{text}</Text>
          {record.is_active === 1 ? (
            <Tag color="green" icon={<CheckCircleOutlined />}>Active</Tag>
          ) : (
            <Tag color="red" icon={<ExclamationCircleOutlined />}>Inactive</Tag>
          )}
        </Space>
      ),
    },
    {
      title: 'Provider',
      dataIndex: 'provider',
      key: 'provider',
      render: (provider) => {
        const colors = {
          openai: 'blue',
          ollama: 'green',
          azure: 'purple',
          anthropic: 'orange',
          local: 'gray'
        };
        return <Tag color={colors[provider]}>{provider.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Base URL',
      dataIndex: 'base_url',
      key: 'base_url',
      ellipsis: true,
      render: (text) => (
        <Tooltip title={text}>
          <Text code>{text}</Text>
        </Tooltip>
      ),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditModel(record)}
          >
            Edit
          </Button>
          <Popconfirm
            title="Are you sure you want to delete this model?"
            onConfirm={() => handleDeleteModel(record.id)}
            okText="Yes"
            cancelText="No"
          >
            <Button
              danger
              size="small"
              icon={<DeleteOutlined />}
            >
              Delete
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Title level={3} style={{ margin: 0 }}>
            <ApiOutlined /> LLM Model Management
          </Title>
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={handleAddModel}
          >
            Add New Model
          </Button>
        </div>

        <Alert
          message="Model Configuration"
          description="Configure LLM models from different providers. Each provider has specific requirements for API keys and endpoints."
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Table
          columns={columns}
          dataSource={models}
          loading={loading}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title={editingModel ? 'Edit LLM Model' : 'Add New LLM Model'}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleModalSubmit}
          initialValues={{
            provider: 'openai',
            is_active: true
          }}
        >
          <Form.Item
            name="provider"
            label="Provider"
            rules={[{ required: true, message: 'Please select a provider' }]}
          >
            <Select
              placeholder="Select LLM Provider"
              onChange={(value) => {
                const config = getProviderConfig(value);
                const defaults = getDefaultModels(value);
                form.setFieldsValue({
                  base_url: config.baseUrl,
                  api_key: '',
                  model_name: defaults[0]?.name || '',
                  description: defaults[0]?.description || ''
                });
              }}
            >
              <Option value="openai">OpenAI</Option>
              <Option value="ollama">Ollama (Local)</Option>
              <Option value="azure">Azure OpenAI</Option>
              <Option value="anthropic">Anthropic Claude</Option>
              <Option value="local">Local Server</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="model_name"
            label="Model Name"
            rules={[{ required: true, message: 'Please enter model name' }]}
          >
            {form.getFieldValue('provider') === 'ollama' ? (
              <Input
                placeholder="Enter custom model name (e.g., llama2, mistral, mixtral)"
                allowClear
              />
            ) : (
              <Select
                placeholder="Select or enter model name"
                showSearch
                allowClear
                notFoundContent="No models found"
              >
                {getDefaultModels(form.getFieldValue('provider') || 'openai').map(model => (
                  <Option key={model.name} value={model.name}>
                    {model.name} - {model.description}
                  </Option>
                ))}
              </Select>
            )}
          </Form.Item>

          <Form.Item
            name="base_url"
            label="Base URL"
            rules={[{ required: true, message: 'Please enter base URL' }]}
          >
            <Input
              placeholder={getProviderConfig(form.getFieldValue('provider') || 'openai').baseUrl}
            />
          </Form.Item>

          <Form.Item
            name="api_key"
            label="API Key"
            help={getProviderConfig(form.getFieldValue('provider') || 'openai').helpText}
          >
            <Input.Password
              placeholder={getProviderConfig(form.getFieldValue('provider') || 'openai').apiKeyPlaceholder}
            />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
            rules={[{ required: true, message: 'Please enter description' }]}
          >
            <TextArea
              rows={3}
              placeholder="Enter model description"
            />
          </Form.Item>

          <Form.Item
            name="is_active"
            label="Status"
            valuePropName="checked"
          >
            <Switch checkedChildren="Active" unCheckedChildren="Inactive" />
          </Form.Item>

          {/* Test Connection Result */}
          {testResult && (
            <Form.Item>
              <Alert
                message={testResult.message}
                type={testResult.success ? 'success' : 'error'}
                description={
                  typeof testResult.details === 'object' 
                    ? JSON.stringify(testResult.details, null, 2)
                    : testResult.details
                }
                showIcon
                style={{ marginBottom: 16 }}
              />
            </Form.Item>
          )}

          <Divider />

          <div style={{ textAlign: 'right' }}>
            <Space>
              <Button onClick={() => setModalVisible(false)}>
                Cancel
              </Button>
              <Button 
                onClick={handleTestConnection}
                loading={testingConnection}
                disabled={testingConnection}
              >
                Test Connection
              </Button>
              <Button type="primary" htmlType="submit">
                {editingModel ? 'Update Model' : 'Create Model'}
              </Button>
            </Space>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default LLMModelManagement;
