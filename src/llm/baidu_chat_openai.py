# baidu_direct_client.py
import requests
import json
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage

class BaiduDirectClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config['openai_api_key']
        self.base_url = config['openai_api_base']
        self.model = config.get('model', '')

    def invoke(self, messages: List[BaseMessage]) -> str:
        """直接调用百度API"""
        # 转换消息格式
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # 处理LangChain消息类型
                role_mapping = {
                    'system': 'user',
                    'human': 'user',
                    'ai': 'assistant'
                }
                formatted_messages.append({
                    'role': role_mapping.get(msg.type, 'user'),
                    'content': msg.content
                })
            else:
                # 兜底处理
                formatted_messages.append({
                    'role': 'user',
                    'content': str(msg)
                })

        payload = json.dumps({
            "model": self.model,
            "messages": formatted_messages,
            "web_search": {
                "enable": False,
                "enable_citation": False,
                "enable_trace": False
            },
            "plugin_options": {}
        }, ensure_ascii=False)

        headers = {
            'Content-Type': 'application/json',
            'appid': '',
            'Authorization': f'Bearer {self.api_key}'
        }

        response = requests.request("POST", self.base_url, headers=headers, data=payload.encode("utf-8"))
        response.encoding = "utf-8"
        result = response.json()
        #print( result)
        return result
