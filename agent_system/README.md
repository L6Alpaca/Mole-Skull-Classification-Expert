# 鼹鼠分类智能体系统

## 概述

本系统是一个功能完整的鼹鼠分类智能体，结合深度学习图像分类与RAG（检索增强生成）技术，实现从图像输入到智能问答的完整流程。

**核心能力：**
- 四视角图像上传与分层分类（属级别 → 种级别）
- 基于EfficientNet-B3的高精度图像分类
- 集成通义千问大语言模型
- 基于Chroma的知识库检索
- 友好的Streamlit Web界面

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       智能体系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐                                              │
│   │   Web界面    │  app.py - Streamlit前端                       │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐     ┌──────────────────────────────────┐    │
│   │   智能体核心  │────▶│         服务层                      │    │
│   │    agent.py   │     │                                  │    │
│   └──────────────┘     │  ┌─────────────────────────────┐  │    │
│                        │  │   image_classifier.py        │  │    │
│                        │  │   - 属级别分类模型            │  │    │
│                        │  │   - 种级别分类模型（7个属）    │  │    │
│                        │  └─────────────────────────────┘  │    │
│                        │                                  │    │
│                        │  ┌─────────────────────────────┐  │    │
│                        │  │   rag_service.py            │  │    │
│                        │  │   - 结合分类结果的问答        │  │    │
│                        │  │   - 流式输出支持              │  │    │
│                        │  └─────────────────────────────┘  │    │
│                        │                                  │    │
│                        │  ┌─────────────────────────────┐  │    │
│                        │  │   knowledge_base.py         │  │    │
│                        │  │   - 知识库管理               │  │    │
│                        │  │   - 向量数据库操作            │  │    │
│                        │  └─────────────────────────────┘  │    │
│                        │                                  │    │
│                        │  ┌─────────────────────────────┐  │    │
│                        │  │   vector_stores.py          │  │    │
│                        │  │   - Chroma向量存储           │  │    │
│                        │  │   - 文档检索功能             │  │    │
│                        │  └─────────────────────────────┘  │    │
│                        │                                  │    │
│                        │  ┌─────────────────────────────┐  │    │
│                        │  │   file_history_store.py     │  │    │
│                        │  │   - 对话历史持久化           │  │    │
│                        │  └─────────────────────────────┘  │    │
│                        └──────────────────────────────────┘    │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      数据层                              │  │
│   │  weights/ (模型权重)  │  chroma_db/ (向量数据库)        │  │
│   │  species_classfier/  │  chat_history/ (对话历史)        │  │
│   │  uploads/ (上传文件)  │  ../data/ (标签文件)            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模块说明

### 1. 配置模块 (`config.py`)

**职责**：集中管理系统所有配置参数

**主要配置项：**

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `GENUS_MODEL_PATH` | 属级别分类模型路径 | `weights/EfficientNet-B3/best_network.pth` |
| `GENUS_LABELS_PATH` | 属级别标签文件 | `data/genus_labels.json` |
| `SPECIES_CLASSIFIER_DIR` | 种级别分类器目录 | `species_classfier/` |
| `SPECIES_LABELS_PATH` | 种级别标签文件 | `species_classfier/data/more_species_labels.json` |
| `POLYTYPIC_GENERA` | 多型属列表（7个属） | Euroscaptor, Mogera, Parascaptor等 |
| `KNOWLEDGE_BASE_DIR` | 知识库目录 | `knowledge_data/` |
| `VECTOR_DB_DIR` | 向量数据库目录 | `chroma_db/` |
| `CHAT_MODEL_NAME` | 大模型名称 | `qwen3-max` |
| `EMBEDDING_MODEL_NAME` | 嵌入模型名称 | `text-embedding-v4` |
| `CHUNK_SIZE` | 文本分割大小 | 1000 |
| `CHUNK_OVERLAP` | 分割重叠长度 | 100 |
| `SIMILARITY_THRESHOLD` | 检索返回数量 | 3 |

---

### 2. 智能体核心 (`agent.py`)

**职责**：系统核心控制器，整合所有服务模块

**核心类：`EuroscaptorAgent`**

| 方法 | 功能 | 参数 | 返回值 |
|------|------|------|--------|
| `__init__()` | 初始化智能体 | 无 | 智能体实例 |
| `upload_images()` | 上传图像到临时目录 | `image_files`: 图像文件列表 | 上传目录路径 |
| `classify()` | 执行分层分类 | `image_dir`: 图像目录（可选） | 分类结果字典 |
| `ask()` | 问答（同步） | `question`: 用户问题 | 回答字符串 |
| `ask_stream()` | 问答（流式） | `question`: 用户问题 | 生成器 |
| `clear_session()` | 清空当前会话 | 无 | 无 |
| `get_class_names()` | 获取所有属类别 | 无 | 类别列表 |
| `get_state()` | 获取当前状态 | 无 | 状态字典 |

**分类结果结构：**
```python
{
    "genus": "Euroscaptor",      # 属名
    "species": "Euroscaptor longirostris",  # 种名（多型属）
    "final_name": "Euroscaptor longirostris",  # 最终分类结果
    "class_idx": 3,              # 类别索引
    "status": "success",         # 状态
    "message": "分类成功"        # 消息
}
```

---

### 3. 图像分类服务 (`image_classifier.py`)

**职责**：实现分层图像分类（属级别 → 种级别）

**核心类：**

#### `ImageClassifierService`
- 加载属级别分类模型（18个属）
- 执行四视角加权融合分类
- 自动调用种级别分类器

#### `SpeciesClassifier`
- 管理7个多型属的种级别分类模型
- 根据属名选择对应的分类器
- 支持 Euroscaptor、Mogera、Parascaptor、Scapanus、Scaptonyx、Talpa、Uropsilus

**多型属种数量：**

| 属名 | 种数量 | 模型路径 |
|------|--------|----------|
| Euroscaptor | 9 | `species_classfier/weights/EB3_Euroscaptor/` |
| Mogera | 9 | `species_classfier/weights/EB3_Mogera/` |
| Parascaptor | 3 | `species_classfier/weights/EB3_Parascaptor/` |
| Scapanus | 4 | `species_classfier/weights/EB3_Scapanus/` |
| Scaptonyx | 3 | `species_classfier/weights/EB3_Scaptonyx/` |
| Talpa | 7 | `species_classfier/weights/EB3_Talpa/` |
| Uropsilus | 5 | `species_classfier/weights/EB3_Uropsilus/` |

**分类流程：**
```
上传四视角图像 → 属级别分类 → 判断是否多型属 → 种级别分类 → 返回结果
```

---

### 4. RAG服务 (`rag_service.py`)

**职责**：结合图像分类结果与知识库进行智能问答

**核心类：`RagService`**

| 方法 | 功能 |
|------|------|
| `set_classification()` | 设置当前分类结果 |
| `ask()` | 同步问答 |
| `ask_stream()` | 流式问答 |
| `clear_history()` | 清空对话历史 |

**检索策略：**
- 优先使用种名进行检索（如 `Euroscaptor longirostris`）
- 未识别到种时使用属名检索（如 `Euroscaptor`）
- 检索词格式：`{物种名/属名} + 用户问题`

**提示词模板：**
```
你是一个专业的鼹鼠分类专家助手。
以下是系统识别出的图像分类结果：{classification_result}
请结合以下参考资料回答用户问题。参考资料:{context}。
```

---

### 5. 知识库服务 (`knowledge_base.py`)

**职责**：管理知识库内容，支持文本向量化存储

**核心类：`KnowledgeBaseService`**

| 方法 | 功能 | 参数 |
|------|------|------|
| `upload_by_str()` | 上传字符串内容 | `data`: 文本内容, `filename`: 文件名 |
| `upload_by_file()` | 上传文件 | `file_path`: 文件路径 |
| `load_directory()` | 批量加载目录 | `dir_path`: 目录路径 |
| `clear_knowledge()` | 清空知识库 | 无 |

**特性：**
- 基于MD5的内容去重
- 自动文本分割（超过1000字符）
- 支持UTF-8编码的txt文件

---

### 6. 向量存储服务 (`vector_stores.py`)

**职责**：封装Chroma向量数据库操作

**核心类：`VectorStoreService`**

| 方法 | 功能 |
|------|------|
| `__init__()` | 初始化向量存储 |
| `get_retriever()` | 返回检索器实例 |

**配置项：**
- `collection_name`: 集合名称（`euroscaptor_rag`）
- `embedding_function`: 嵌入函数（DashScopeEmbeddings）
- `persist_directory`: 持久化目录

---

### 7. 对话历史存储 (`file_history_store.py`)

**职责**：管理对话历史的持久化存储

**核心类：`FileChatMessageHistory`**

| 方法 | 功能 |
|------|------|
| `add_message()` | 添加消息 |
| `clear()` | 清空历史 |
| `messages` | 获取消息列表（属性） |

**存储格式：**
- 文件路径：`chat_history/{session_id}.json`
- 使用LangChain的消息序列化格式

---

### 8. Web界面 (`app.py`)

**职责**：提供用户友好的交互界面

**界面结构：**

| 区域 | 功能 |
|------|------|
| 侧边栏 | 图像上传、预览、识别按钮、清空会话 |
| 主区域 | 对话历史、输入框 |
| 状态显示 | 当前分类结果、支持的类别列表 |

**使用流程：**
1. 上传四视角图像（支持 jpg, jpeg, png）
2. 点击"开始识别"按钮
3. 查看识别结果（属名和种名）
4. 在对话区域提问

---

## 目录结构

```
agent_system/
├── config.py              # 配置文件
├── agent.py               # 智能体核心模块
├── app.py                 # Streamlit主界面
├── image_classifier.py    # 图像分类服务
├── rag_service.py         # RAG问答服务
├── knowledge_base.py      # 知识库服务
├── vector_stores.py       # 向量存储服务
├── file_history_store.py  # 对话历史存储
├── requirements.txt       # 依赖包列表
├── README.md              # 本文件
├── GETTING_STARTED.md     # 快速开始指南
├── run_ai.bat             # Windows启动脚本
├── start_in_ai_env.bat    # Conda环境启动脚本
├── chroma_db/             # 向量数据库目录
├── chat_history/          # 对话历史目录
└── uploads/               # 上传文件目录
```

---

## 快速开始

### 环境要求
- Python 3.8+
- Conda环境（推荐使用 `ai` 环境）
- DashScope API Key

### 安装步骤

1. **激活环境**
```bash
conda activate ai
```

2. **安装依赖**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit langchain langchain-community langchain-chroma chromadb dashscope numpy Pillow python-multipart
```

3. **设置API Key**
```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

4. **启动应用**
```bash
streamlit run app.py
```

或使用启动脚本：
```bash
run_ai.bat
```

---

## 知识库扩展

### 添加新内容

1. **方式一：添加txt文件**
   - 将知识内容保存为 `.txt` 文件
   - 放入 `knowledge_data/` 目录
   - 重启应用自动加载

2. **方式二：使用API**
```python
from knowledge_base import KnowledgeBaseService

kb = KnowledgeBaseService()

# 上传字符串
kb.upload_by_str("物种描述内容...", "filename.txt")

# 上传文件
kb.upload_by_file("path/to/file.txt")

# 批量加载目录
kb.load_directory("path/to/directory/")
```

### 文件格式建议

```
物种名称：长吻鼹（Euroscaptor longirostris）

形态特征：
- 吻部细长，吻端裸露
- 身体呈圆筒形，四肢短粗
- 前足掌宽大，具有强大的爪

栖息环境：
- 主要栖息于海拔1000-3000米的森林地带
- 喜欢湿润的土壤环境

地理分布：
- 中国特有种
- 分布于四川、云南、贵州等地
```

---

## 技术栈

| 类别 | 技术 | 版本要求 |
|------|------|----------|
| 深度学习框架 | PyTorch | 2.0+ |
| 模型架构 | EfficientNet-B3 | - |
| Web框架 | Streamlit | 1.30+ |
| 大模型 | 通义千问 | DashScope |
| 向量数据库 | Chroma | 0.4+ |
| LLM框架 | LangChain | 0.1+ |
| 嵌入模型 | DashScope Embedding | text-embedding-v4 |

---

## 常见问题

### 1. 模型加载失败
- 检查模型路径是否正确
- 确保 `weights/EfficientNet-B3/best_network.pth` 存在

### 2. 分类结果不准确
- 确保上传完整的四视角图像
- 图像文件名需包含视角标记（s#v, s#d, s#l, m#l）

### 3. API Key错误
- 确保设置了 `DASHSCOPE_API_KEY` 环境变量
- 检查API Key是否有效

### 4. Streamlit启动失败
- 使用 `python -m streamlit run app.py` 命令
- 确保端口未被占用（默认8501）

---

## 维护说明

### 日志与调试

系统启动时会输出各服务的初始化状态：
- ✅ 图像分类服务初始化成功
- ✅ 知识库服务初始化成功
- ✅ RAG服务初始化成功

### 清理临时文件

上传的图像文件保存在 `uploads/` 目录，定期清理：
```bash
rm -rf uploads/*
```

### 重新构建向量数据库

如需重新加载知识库：
```python
from knowledge_base import KnowledgeBaseService

kb = KnowledgeBaseService()
kb.clear_knowledge()
kb.load_directory("path/to/knowledge_base/")
```

---

## 许可证

本项目仅供内部研究使用。
