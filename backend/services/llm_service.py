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
        prompt: str,
        model_config: Dict,
        context: Optional[str] = None
    ) -> str:
        """
        Generate response using configured LLM
        
        Args:
            prompt: User prompt/question
            model_config: Model configuration dict with provider, model_name, base_url, etc.
            context: Optional context/retrieved documents
            
        Returns:
            Generated response text
        """
        provider = model_config.get('provider')
        model_name = model_config.get('model_name')
        base_url = model_config.get('base_url')
        
        # Build prompt with context if provided
        if context:
            system_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
If the context doesn't contain relevant information, say so.

Context:
{context}

User Question: {prompt}

Answer:"""
        else:
            system_prompt = prompt
        
        if provider == 'ollama':
            messages = [
                {"role": "user", "content": system_prompt}
            ]
            result = self.chat_with_ollama(model_name, messages, base_url)
            return result.get('content', '')
        else:
            raise NotImplementedError(f"Provider '{provider}' not yet implemented")

