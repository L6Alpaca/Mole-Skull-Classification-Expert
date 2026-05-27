# 项目概览 | Project Overview

## 快速导航

- 📖 [主 README](README.md) - 项目介绍、安装、使用
- 🤝 [贡献指南](CONTRIBUTING.md) - 如何贡献代码
- 📄 [许可证](LICENSE) - MIT 许可证
- 📂 [智能体系统文档](agent_system/README.md) - 详细技术文档

## 项目结构

```
Euroscaptor/
├── 📄 README.md              # 主文档
├── 📄 CONTRIBUTING.md        # 贡献指南
├── 📄 LICENSE                # 许可证
├── 📄 .gitignore             # Git 忽略配置
│
├── 📂 agent_system/          # 智能体系统核心
│   ├── config.py            # 配置文件
│   ├── agent.py             # 智能体核心
│   ├── app.py               # Web 界面
│   ├── image_classifier.py  # 图像分类服务
│   ├── rag_service.py       # RAG 问答服务
│   ├── knowledge_base.py    # 知识库服务
│   ├── vector_stores.py     # 向量存储
│   ├── file_history_store.py # 对话历史
│   ├── knowledge_data/      # 物种知识库
│   ├── requirements.txt     # Python 依赖
│   └── README.md            # 系统详细文档
│
├── 📂 weights/               # 模型权重（需单独下载）
│   └── EfficientNet-B3/     # 属级别分类模型
│
├── 📂 species_classfier/     # 种级别分类器
│   └── weights/             # 7 个属的种级模型
│
└── 📂 data/                  # 数据文件
    ├── genus_labels.json    # 属标签映射
    └── species_labels.json  # 种标签映射
```

## 核心功能模块

### 1. 图像分类模块 (`image_classifier.py`)

**功能**：对鼹鼠四视角图像进行分层分类

**关键类**：
- `ImageClassifierService` - 图像分类服务
- `SpeciesClassifier` - 种级别分类器

**使用示例**：
```python
from image_classifier import ImageClassifierService

classifier = ImageClassifierService()
result = classifier.classify("path/to/images/")
print(f"识别结果：{result['final_name']}")
```

### 2. RAG 问答模块 (`rag_service.py`)

**功能**：基于识别结果和知识库进行智能问答

**关键类**：
- `RagService` - RAG 问答服务

**使用示例**：
```python
from rag_service import RagService

rag = RagService()
rag.set_classification(classification_result)
answer = rag.ask("这个物种的栖息环境是什么？")
print(answer)
```

### 3. 知识库模块 (`knowledge_base.py`)

**功能**：管理知识库的加载、存储和检索

**关键类**：
- `KnowledgeBaseService` - 知识库服务

**使用示例**：
```python
from knowledge_base import KnowledgeBaseService

kb = KnowledgeBaseService()
kb.load_directory("path/to/knowledge/files/")
```

### 4. 智能体核心 (`agent.py`)

**功能**：整合图像分类和 RAG 问答，提供统一接口

**关键类**：
- `EuroscaptorAgent` - 智能体主类

**使用示例**：
```python
from agent import get_agent

agent = get_agent()
upload_dir = agent.upload_images(image_files)
result = agent.classify(upload_dir)
answer = agent.ask("这个物种有什么特征？")
```

### 5. Web 界面 (`app.py`)

**功能**：基于 Streamlit 的交互式 Web 界面

**启动方式**：
```bash
cd agent_system
streamlit run app.py
```

## 数据流

```
用户上传图像
    ↓
图像分类服务（属级 → 种级）
    ↓
分类结果存储
    ↓
用户提问
    ↓
RAG 服务（检索 + 生成）
    ↓
返回答案
```

## 技术架构图

```
┌─────────────────────────────────────────┐
│         Streamlit Web 界面               │
│         (app.py)                        │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         EuroscaptorAgent                │
│         (agent.py)                      │
└───────┬──────────────────┬──────────────┘
        │                  │
┌───────▼──────────┐  ┌───▼──────────────────────┐
│ ImageClassifier  │  │   RagService             │
│ Service          │  │   (rag_service.py)       │
│ (image_classifier│  │   - 检索 (Chroma)        │
│  .py)            │  │   - 生成 (LLM)           │
│ - 属级分类       │  │                          │
│ - 种级分类       │  └───────────┬──────────────┘
└──────────────────┘              │
                              ┌───▼──────────────────────┐
                              │   KnowledgeBaseService   │
                              │   (knowledge_base.py)    │
                              └──────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 创建 Conda 环境
conda create -n mole_agent python=3.11
conda activate mole_agent

# 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r agent_system/requirements.txt
```

### 2. 配置 API Key

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 启动应用

```bash
cd agent_system
streamlit run app.py
```

## 常用命令

```bash
# 运行测试
pytest tests/

# 代码格式化
black agent_system/*.py

# 检查代码风格
flake8 agent_system/*.py

# 查看依赖
pip list

# 更新依赖
pip install -r agent_system/requirements.txt --upgrade
```

## 目录说明

| 目录 | 用途 | 是否提交到 Git |
|------|------|---------------|
| `agent_system/` | 核心代码 | ✅ 是 |
| `weights/` | 模型权重 | ⚠️ 大文件需单独下载 |
| `species_classfier/` | 种级分类器 | ⚠️ 大文件需单独下载 |
| `data/` | 标签数据 | ✅ 是 |
| `大模型案例/` | 知识库 | ✅ 是 |
| `agent_system/uploads/` | 上传文件 | ❌ 否（.gitignore） |
| `agent_system/chat_history/` | 对话历史 | ❌ 否（.gitignore） |
| `agent_system/chroma_db/` | 向量数据库 | ❌ 否（可重建） |

## 依赖说明

### 核心依赖

- **PyTorch**: 深度学习框架
- **EfficientNet-B3**: 图像分类模型
- **Streamlit**: Web 界面框架
- **LangChain**: LLM 应用框架
- **Chroma**: 向量数据库
- **DashScope**: 通义千问 API

### 可选依赖

- **CUDA**: GPU 加速（如需要）
- **Black/Flake8**: 代码格式化和检查
- **Pytest**: 单元测试

## 常见问题

### Q: 模型权重文件在哪里？

A: 由于文件较大，权重文件未包含在 Git 仓库中。请联系作者获取或从指定下载地址获取。

### Q: 如何添加新的知识库内容？

A: 将文本文件放入 `大模型案例/Euroscaptor_data/` 目录，系统会自动加载。

### Q: 如何修改分类阈值？

A: 在 `config.py` 中修改 `CONFIDENCE_THRESHOLD` 参数。

### Q: 支持哪些图像格式？

A: 支持 JPG、JPEG、PNG 等常见格式。

## 开发路线图

### v1.0 (当前版本)
- ✅ 属级别分类（18 属）
- ✅ 种级别分类（7 个多型属）
- ✅ RAG 问答系统
- ✅ Web 界面

### v1.1 (计划)
- [ ] 支持更多属的种级分类
- [ ] 批量图像处理
- [ ] 导出分类报告

### v2.0 (规划)
- [ ] 模型微调支持
- [ ] 多语言界面
- [ ] API 服务化

## 贡献者

感谢所有为本项目做出贡献的开发者！

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

<div align="center">

**开始您的探索之旅吧！** 🚀

[返回主 README](README.md)

</div>
