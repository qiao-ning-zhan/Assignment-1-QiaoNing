import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIInterface:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _send_request(self, endpoint: str, payload: Dict) -> Dict:
        response = requests.post(f"{self.base_url}/{endpoint}", headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def generate_chat_response(self, conversation: List[Dict[str, str]]) -> str:
        payload = {
            "model": "gpt-4-mini",
            "messages": conversation
        }
        response = self._send_request("chat/completions", payload)
        return response['choices'][0]['message']['content']

    def create_embedding(self, text: str) -> List[float]:
        payload = {
            "model": "text-embedding-3-large",
            "input": text
        }
        response = self._send_request("embeddings", payload)
        return response['data'][0]['embedding']