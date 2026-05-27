"""
智能体系统配置文件
"""
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 模型路径
GENUS_MODEL_PATH = str(BASE_DIR / "weights" / "EfficientNet-B3" / "best_network.pth")
GENUS_LABELS_PATH = str(BASE_DIR / "data" / "genus_labels.json")

# 种级别分类器配置
SPECIES_CLASSIFIER_DIR = str(BASE_DIR / "species_classfier")
SPECIES_LABELS_PATH = str(BASE_DIR / "species_classfier" / "data" / "more_species_labels.json")

# 多型属列表（需要进行种级别分类的属）
POLYTYPIC_GENERA = [
    "Euroscaptor",
    "Mogera", 
    "Parascaptor",
    "Scapanus",
    "Scaptonyx",
    "Talpa",
    "Uropsilus"
]

# 知识库路径
KNOWLEDGE_BASE_DIR = str(BASE_DIR / "agent_system" / "knowledge_data")
VECTOR_DB_DIR = str(BASE_DIR / "agent_system" / "chroma_db")
MD5_PATH = str(BASE_DIR / "agent_system" / "md5.text")

# 大模型配置
EMBEDDING_MODEL_NAME = "text-embedding-v4"
CHAT_MODEL_NAME = "qwen3-max"
COLLECTION_NAME = "euroscaptor_rag"

# 文本分割配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
MAX_SPLIT_CHAR_NUMBER = 1000

# 检索配置
SIMILARITY_THRESHOLD = 3  # 检索返回匹配的文档数量

# 图像上传配置
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
UPLOAD_DIR = str(BASE_DIR / "agent_system" / "uploads")

# Session配置
SESSION_CONFIG = {
    "configurable": {
        "session_id": "euroscaptor_user_001",
    }
}
