"""
知识库服务模块
"""
import os
import hashlib
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def check_md5(md5_str: str):
    """检查传入的md5字符串是否已经被处理过了
        return False(md5未处理过)  True(已经处理过，已有记录）
    """
    if not os.path.exists(config.MD5_PATH):
        open(config.MD5_PATH, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(config.MD5_PATH, 'r', encoding='utf-8').readlines():
            line = line.strip()
            if line == md5_str:
                return True
        return False


def save_md5(md5_str: str):
    """将传入的md5字符串，记录到文件内保存"""
    with open(config.MD5_PATH, 'a', encoding="utf-8") as f:
        f.write(md5_str + '\n')


def get_string_md5(input_str: str, encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex


class KnowledgeBaseService:
    """知识库服务"""

    def __init__(self):
        os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)

        self.chroma = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=DashScopeEmbeddings(model=config.EMBEDDING_MODEL_NAME),
            persist_directory=config.VECTOR_DB_DIR,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )

    def upload_by_str(self, data: str, filename):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过]内容已经存在知识库中"

        if len(data) > config.MAX_SPLIT_CHAR_NUMBER:
            knowledge_chunks = self.spliter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "agent_system",
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        save_md5(md5_hex)
        return "[成功]内容已经成功载入向量库"

    def upload_by_file(self, file_path):
        """从文件上传知识库内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
            filename = os.path.basename(file_path)
            return self.upload_by_str(data, filename)
        except Exception as e:
            return f"[失败]文件上传失败: {str(e)}"

    def load_directory(self, dir_path):
        """批量加载目录下的所有txt文件到知识库"""
        results = []
        if not os.path.exists(dir_path):
            results.append(f"[警告]目录不存在: {dir_path}")
            return results

        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                result = self.upload_by_file(file_path)
                results.append(f"{filename}: {result}")
        return results

    def clear_knowledge(self):
        """清空知识库"""
        try:
            self.chroma.delete_collection()
            self.chroma = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=DashScopeEmbeddings(model=config.EMBEDDING_MODEL_NAME),
                persist_directory=config.VECTOR_DB_DIR,
            )
            if os.path.exists(config.MD5_PATH):
                os.remove(config.MD5_PATH)
            return "知识库已清空"
        except Exception as e:
            return f"清空失败: {str(e)}"


if __name__ == "__main__":
    print("初始化知识库服务...")
    kb_service = KnowledgeBaseService()

    print("加载现有的Euroscaptor知识库...")
    results = kb_service.load_directory(config.KNOWLEDGE_BASE_DIR)
    for r in results:
        print(r)
