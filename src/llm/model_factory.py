import httpx
import yaml
import os
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from typing import Dict, Any

from src.llm.baidu_chat_openai import BaiduDirectClient


class ModelFactory:
    def __init__(self, config_path: str):
        """
        初始化模型工厂

        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    @staticmethod
    def get_project_config_path(config_file_name: str = 'models.yaml') -> str:
        """
        获取项目配置文件路径

        Args:
            config_file_name: 配置文件名

        Returns:
            配置文件完整路径
        """
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        return os.path.join(project_root, 'config', config_file_name)

    def create_model(self, model_name: str) -> Any:
        """
        根据模型名称创建模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例
        """

        if model_name not in self.config['models']:
            raise ValueError(f"Model {model_name} not found in config")

        model_config = self.config['models'][model_name]
        model_type = model_config['type']

        if model_type == 'openai':
            return self._create_openai_model(model_config)
        if model_type == 'ollama':
            return self._create_ollama_model(model_config)
        if model_type == 'baiduOld':
            print("使用baiduOld模型")
            return BaiduDirectClient(model_config)
        elif model_type == 'custom':
            return self._create_custom_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _create_openai_model(config: Dict[str, Any]) -> ChatOpenAI:
        """
        创建OpenAI兼容模型实例
        """
        return ChatOpenAI(
            base_url=config['openai_api_base'],
            model=config['model'],
            api_key=config['openai_api_key'],
            http_client=httpx.Client(proxy=config.get('proxy'))  # 添加代理支持
        )

    @staticmethod
    def _create_ollama_model(config: Dict[str, Any]) -> OllamaLLM:
        """
        创建OpenAI兼容模型实例
        """
        return OllamaLLM(
            base_url=config.get('base_url', 'http://localhost:11434'),
            model=config['model']
        )

    def _create_custom_model(self, config: Dict[str, Any]) -> Any:
        """
        创建自定义模型实例（根据实际需求实现）
        """
        # 这里可以实现自定义模型的创建逻辑
        pass

if __name__ == '__main__':
    # 使用模型工厂
    #from src.llm.model_factory import ModelFactory

    import time
    import os
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_file_dir))
    print(project_root)
    file_path = os.path.join(project_root, 'config', 'models.yaml')
    # 初始化模型工厂
    factory = ModelFactory(file_path)

    start_time = time.time()
    # 根据名称创建模型
    model = factory.create_model('local-qwen3-8b')
    #model = factory.create_model('qwen3-30b')
    #model = factory.create_model('gpt-4')
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    # 使用模型进行推理
    response = model.invoke(messages)
    print (response)

    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"处理时间: {processing_time:.2f} 秒")