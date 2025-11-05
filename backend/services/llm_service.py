"""
LLM Service
Handles LLM interactions with Ollama and other providers
"""

import requests
from typing import Dict, List, Optional
import json


class LLMService:
    """Service for interacting with LLM models"""
    
    def __init__(self):
        self.ollama_base_url = "http://localhost:11434"
    
    def chat_with_ollama(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        base_url: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """
        Chat with Ollama model
        
        Args:
            model_name: Name of the Ollama model
            messages: List of message dicts with 'role' and 'content'
            base_url: Optional base URL for Ollama (default: localhost:11434)
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response content
        """
        url = f"{base_url or self.ollama_base_url}/api/chat"
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": stream
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                return self._handle_stream_response(response)
            else:
                result = response.json()
                return {
                    'content': result.get('message', {}).get('content', ''),
                    'model': result.get('model', model_name),
                    'done': result.get('done', True)
                }
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama server at {base_url or self.ollama_base_url}")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out")
        except Exception as e:
            raise Exception(f"Failed to chat with Ollama: {str(e)}")
    
    def _handle_stream_response(self, response):
        """Handle streaming response from Ollama"""
        full_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    content = chunk.get('message', {}).get('content', '')
                    full_content += content
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return {
            'content': full_content,
            'stream': True
        }
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model_config: Optional[Dict] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using configured LLM with messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_config: Optional model configuration dict (for backward compatibility)
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        # Check if messages is a string (backward compatibility)
        if isinstance(messages, str):
            prompt = messages
            model_name = model_config.get('model_name') if model_config else None
            base_url = model_config.get('base_url') if model_config else None
            provider = model_config.get('provider') if model_config else 'ollama'
            
            messages = [{"role": "user", "content": prompt}]
        else:
            # Extract model config from first message or use default
            provider = 'ollama'
            model_name = None
            base_url = None
            
            if model_config:
                provider = model_config.get('provider', 'ollama')
                model_name = model_config.get('model_name')
                base_url = model_config.get('base_url')
        
        if provider == 'ollama':
            # Use default if not provided
            if not model_name:
                model_name = "llama2"  # Default Ollama model
            if not base_url:
                base_url = self.ollama_base_url
            
            result = self.chat_with_ollama(model_name, messages, base_url)
            return result.get('content', '')
        elif provider == 'openai':
            # OpenAI API support
            import os
            api_key = model_config.get('api_key') if model_config else os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key is required. Set it in model_config or OPENAI_API_KEY environment variable.")
            
            base_url = model_config.get('base_url') if model_config else 'https://api.openai.com/v1'
            model_name = model_config.get('model_name') if model_config else 'gpt-3.5-turbo'
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        elif provider == 'anthropic':
            # Anthropic Claude API support
            import os
            api_key = model_config.get('api_key') if model_config else os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key is required. Set it in model_config or ANTHROPIC_API_KEY environment variable.")
            
            base_url = model_config.get('base_url') if model_config else 'https://api.anthropic.com/v1'
            model_name = model_config.get('model_name') if model_config else 'claude-3-sonnet-20240229'
            
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            # Convert messages format for Anthropic (system message separate)
            system_message = None
            user_messages = []
            for msg in messages:
                if msg.get('role') == 'system':
                    system_message = msg.get('content', '')
                else:
                    user_messages.append(msg)
            
            payload = {
                "model": model_name,
                "max_tokens": 4096,
                "temperature": temperature,
                "messages": user_messages
            }
            
            if system_message:
                payload["system"] = system_message
            
            response = requests.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text']
        else:
            raise NotImplementedError(f"Provider '{provider}' not yet implemented. Supported providers: ollama, openai, anthropic")

