"""
Test script to verify embedding models endpoint
"""
import requests
import json

try:
    response = requests.get('http://localhost:8000/api/embedding-models')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print(f"Response Type: {type(response.json())}")
    print(f"Response Length: {len(response.json()) if isinstance(response.json(), list) else 'Not a list'}")
except Exception as e:
    print(f"Error: {str(e)}")

