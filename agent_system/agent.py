"""
智能体核心服务模块
整合图像分类和RAG问答功能
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

import config
from image_classifier import ImageClassifierService
from knowledge_base import KnowledgeBaseService
from rag_service import RagService


class EuroscaptorAgent:
    """
    鼹鼠分类智能体
    """

    def __init__(self):
        """
        初始化智能体
        """
        # 创建必要的目录
        os.makedirs(config.UPLOAD_DIR, exist_ok=True)

        # 初始化服务
        self.image_classifier = None
        self.kb_service = None
        self.rag_service = None

        # 当前状态
        self.current_classification = None
        self.current_upload_dir = None

        self._init_services()

    def _init_services(self):
        """初始化所有服务"""
        print("正在初始化智能体服务...")
        try:
            self.image_classifier = ImageClassifierService()
            print("✅ 图像分类服务初始化成功")
        except Exception as e:
            print(f"❌ 图像分类服务初始化失败: {e}")

        try:
            self.kb_service = KnowledgeBaseService()
            print("✅ 知识库服务初始化成功")
            # 自动加载知识库
            results = self.kb_service.load_directory(config.KNOWLEDGE_BASE_DIR)
            for r in results:
                print(f"   {r}")
        except Exception as e:
            print(f"❌ 知识库服务初始化失败: {e}")

        try:
            self.rag_service = RagService()
            print("✅ RAG服务初始化成功")
        except Exception as e:
            print(f"❌ RAG服务初始化失败: {e}")

        print("智能体初始化完成！")

    def upload_images(self, image_files: List, temp_dir: Optional[str] = None) -> str:
        """
        上传四视角图像
        :param image_files: 图像文件列表
        :param temp_dir: 临时目录，如果不提供会自动生成
        :return: 上传目录路径
        """
        # 生成临时目录
        if temp_dir is None:
            temp_dir = os.path.join(config.UPLOAD_DIR, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        # 保存文件
        saved_files = []
        for i, img_file in enumerate(image_files):
            # 如果是streamlit上传的文件
            if hasattr(img_file, 'name') and hasattr(img_file, 'getvalue'):
                file_name = img_file.name
                file_data = img_file.getvalue()
            else:
                # 其他类型的文件对象
                file_name = f"image_{i}.jpg"
                file_data = img_file

            # 保存文件
            file_path = os.path.join(temp_dir, file_name)
            if isinstance(file_data, bytes):
                with open(file_path, 'wb') as f:
                    f.write(file_data)
            else:
                shutil.copy(file_data, file_path)

            saved_files.append(file_path)

        self.current_upload_dir = temp_dir
        print(f"✅ 已上传 {len(saved_files)} 张图像到 {temp_dir}")
        return temp_dir

    def classify(self, image_dir: Optional[str] = None) -> Dict:
        """
        执行分类
        :param image_dir: 图像目录，如果不提供使用当前上传目录
        :return: 分类结果
        """
        if image_dir is None:
            image_dir = self.current_upload_dir

        if image_dir is None or not os.path.exists(image_dir):
            return {
                "status": "error",
                "message": "请先上传图像"
            }

        if self.image_classifier is None:
            return {
                "status": "error",
                "message": "图像分类服务未初始化"
            }

        result = self.image_classifier.classify(image_dir)
        self.current_classification = result

        # 更新RAG服务的分类结果
        if self.rag_service:
            self.rag_service.set_classification(result)

        return result

    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        """
        提问
        :param question: 用户问题
        :param session_id: 会话ID
        :return: AI回答
        """
        if self.rag_service is None:
            return "RAG服务未初始化"

        return self.rag_service.ask(question, session_id)

    def ask_stream(self, question: str, session_id: Optional[str] = None):
        """
        流式提问
        :param question: 用户问题
        :param session_id: 会话ID
        :yield: AI回答片段
        """
        if self.rag_service is None:
            yield "RAG服务未初始化"
            return

        yield from self.rag_service.ask_stream(question, session_id)

    def clear_session(self):
        """清空当前会话状态"""
        self.current_classification = None
        self.current_upload_dir = None

        if self.rag_service:
            self.rag_service.set_classification(None)

        print("会话状态已清空")

    def get_class_names(self) -> List[str]:
        """获取所有分类类别"""
        if self.image_classifier:
            return self.image_classifier.get_class_names()
        return []

    def get_state(self) -> Dict:
        """获取当前状态"""
        return {
            "classification": self.current_classification,
            "upload_dir": self.current_upload_dir,
            "class_names": self.get_class_names()
        }


# 全局单例
_agent_instance = None


def get_agent() -> EuroscaptorAgent:
    """获取智能体单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = EuroscaptorAgent()
    return _agent_instance


if __name__ == "__main__":
    print("测试智能体...")
    agent = get_agent()

    print(f"支持的类别: {agent.get_class_names()}")
