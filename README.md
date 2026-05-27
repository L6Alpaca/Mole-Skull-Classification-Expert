# 鼹鼠分类智能体系统 | Mole Classification Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io)

一个融合深度学习与大型语言模型的多模态生物分类智能体系统，实现鼹鼠物种的自动化识别与智能问答。

## 🌟 项目亮点

- **分层识别架构**：属级别→种级别两级分类，支持 18 个属、40+ 物种的精细识别
- **多模态融合**：计算机视觉（EfficientNet-B3）+ 自然语言处理（通义千问）
- **RAG 增强问答**：基于向量检索的上下文感知智能问答系统
- **四视角加权**：创新性的多视角图像融合策略，提升分类准确率
- **端到端系统**：从图像上传到智能问答的完整工作流

## 📋 目录

- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [使用方法](#-使用方法)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)
- [数据集](#-数据集)
- [API 参考](#-api-参考)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)
- [引用](#-引用)
- [联系方式](#-联系方式)

## ✨ 功能特性

### 核心功能

1. **图像分类**
   - 支持四视角图像上传（背视、腹视、侧视、 mandible 视）
   - 属级别分类：18 个鼹鼠属
   - 种级别分类：7 个多型属的 40+ 物种
   - 基于视角准确率的动态权重融合

2. **智能问答**
   - 基于识别结果的上下文感知问答
   - 物种知识库检索（形态特征、栖息环境、地理分布）
   - 流式输出支持
   - 多轮对话历史维护

3. **知识库管理**
   - 向量数据库（Chroma）存储
   - 自动文本分割与去重
   - 支持批量加载与增量更新

### 技术优势

- **分层分类策略**：先属后种的层级化识别，降低分类复杂度
- **检索优化**：优先使用种名检索，提升问答相关性
- **模块化设计**：清晰的服务层划分，易于扩展与维护
- **用户友好**：基于 Streamlit 的交互式 Web 界面

## ️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web 界面 (Streamlit)                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              智能体核心 (EuroscaptorAgent)               │
│  - 图像上传管理                                          │
│  - 分类结果维护                                          │
│  - 问答请求路由                                          │
──────┬──────────────────────┬──────────────────────────┘
       │                      │
┌──────▼───────────┐  ┌──────▼───────────────────────────┐
│  图像分类服务     │  │       RAG 问答服务                │
│  - 属级模型 (18)  │  │  - 分类结果格式化                 │
│  - 种级模型 (7)   │  │  - 向量检索 (Chroma)             │
│  - 四视角融合     │  │  - LLM 生成 (通义千问)            │
└──────────────────┘  └───────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Conda（推荐）
- GPU（可选，用于加速推理）
- DashScope API Key（通义千问）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/mole-classification-agent.git
cd mole-classification-agent
```

#### 2. 创建 Conda 环境

```bash
conda create -n mole_agent python=3.11
conda activate mole_agent
```

#### 3. 安装依赖

```bash
# 安装 PyTorch（CPU 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或使用 GPU 版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r agent_system/requirements.txt
```

#### 4. 配置 API Key

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

#### 5. 启动应用

```bash
cd agent_system
streamlit run app.py
```

或使用启动脚本：
```bash
run_ai.bat
```

然后在浏览器中打开 `http://localhost:8501`

## 📖 使用方法

### 基本流程

1. **上传图像**
   - 在侧边栏点击"上传图像"按钮
   - 选择 4 个不同视角的鼹鼠图像
   - 支持格式：JPG, JPEG, PNG

2. **开始识别**
   - 点击"开始识别"按钮
   - 等待分类完成（约 5-10 秒）
   - 查看识别结果（属名 + 种名）

3. **智能问答**
   - 在主界面输入问题
   - 系统结合识别结果和知识库回答
   - 支持连续提问

### 示例问题

- "这个物种的栖息环境是什么？"
- "它有哪些形态特征？"
- "分布于哪些地区？"
- "与相近物种有什么区别？"

### 知识库扩展

添加新的知识内容：

```bash
# 方式 1：直接添加 txt 文件
# 将知识内容保存为.txt 文件，放入 agent_system/knowledge_data/ 目录

# 方式 2：使用 API
python -c "
from knowledge_base import KnowledgeBaseService
kb = KnowledgeBaseService()
kb.upload_by_file('path/to/your/knowledge.txt')
"
```

## 📁 项目结构

```
mole-classification-agent/
├── README.md                    # 项目文档
├── LICENSE                      # 许可证
├── .gitignore                   # Git 忽略文件
├── agent_system/                # 智能体系统
│   ├── config.py               # 配置文件
│   ├── agent.py                # 智能体核心
│   ├── app.py                  # Web 界面
│   ├── image_classifier.py     # 图像分类服务
│   ├── rag_service.py          # RAG 问答服务
│   ├── knowledge_base.py       # 知识库服务
│   ├── vector_stores.py        # 向量存储
│   ├── file_history_store.py   # 对话历史
│   ├── knowledge_data/         # 物种知识库
│   ├── requirements.txt        # Python 依赖
│   └── ...
├── weights/                     # 模型权重
│   └── EfficientNet-B3/        # 属级别模型
├── species_classfier/           # 种级别分类器
│   └── weights/                # 7 个属的种级模型
└── data/                        # 数据文件
    ├── genus_labels.json       # 属标签
    └── species_labels.json     # 种标签
```

## 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| **深度学习** | PyTorch | 2.0+ |
| **模型架构** | EfficientNet-B3 | - |
| **Web 框架** | Streamlit | 1.30+ |
| **大模型** | 通义千问 (DashScope) | qwen3-max |
| **向量数据库** | Chroma | 0.4+ |
| **LLM 框架** | LangChain | 0.1+ |
| **嵌入模型** | DashScope Embedding | text-embedding-v4 |

## 📊 数据集

### 支持分类的属（18 个）

- Euroscaptor（鼹属）
- Mogera（日本鼹属）
- Parascaptor（美洲鼹属）
- Scapanus（北美鼹属）
- Scaptonyx（长尾鼹属）
- Talpa（欧鼹属）
- Uropsilus（缺齿鼹属）
- 等 11 个其他属

### 种级别分类（7 个多型属，40+ 物种）

| 属名 | 物种数 | 示例物种 |
|------|--------|----------|
| Euroscaptor | 9 | E. longirostris, E. micrura |
| Mogera | 9 | M. etigo, M. wogura |
| Parascaptor | 3 | P. leucura |
| Scapanus | 4 | S. latimanus |
| Scaptonyx | 3 | S. fusicaudus |
| Talpa | 7 | T. europaea, T. altaica |
| Uropsilus | 5 | U. soricipes |

## 🔌 API 参考

### 图像分类服务

```python
from image_classifier import ImageClassifierService

# 初始化服务
classifier = ImageClassifierService()

# 执行分类
result = classifier.classify(image_dir="path/to/images/")
print(result)
# 输出:
# {
#   "genus": "Euroscaptor",
#   "species": "Euroscaptor longirostris",
#   "final_name": "Euroscaptor longirostris",
#   "status": "success"
# }
```

### RAG 问答服务

```python
from rag_service import RagService

# 初始化服务
rag = RagService()

# 设置分类结果
rag.set_classification(result)

# 提问
response = rag.ask("这个物种的栖息环境是什么？")
print(response)
```

### 智能体核心

```python
from agent import get_agent

# 获取智能体实例
agent = get_agent()

# 上传图像
upload_dir = agent.upload_images(image_files)

# 分类
result = agent.classify(upload_dir)

# 问答
answer = agent.ask("这个物种有什么特征？")
```

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

详细贡献指南请参考 [CONTRIBUTING.md](CONTRIBUTING.md)

### 代码规范

- 遵循 PEP 8 代码规范
- 使用有意义的变量名和函数名
- 添加必要的文档字符串
- 编写单元测试

### 测试

```bash
# 运行单元测试
python -m pytest tests/

# 运行集成测试
python agent_system/core_test.py
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📚 引用

如果您在研究中使用了本系统，请引用：

```bibtex
@software{mole_classification_agent,
  title = {Mole Classification Agent System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/mole-classification-agent}
}
```

##  联系方式

- **作者**：L6Alpaca
- **邮箱**：1210119689@qq.com
- **问题反馈**：请在 GitHub 提交 Issue

## 🙏 致谢

感谢以下项目与数据提供者：

- EfficientNet 模型作者
- LangChain 团队
- Streamlit 团队
- 标本数据提供机构

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star 支持一下！**

[️ 返回顶部](#鼹鼠分类智能体系统--mole-classification-agent-system)

</div>
# Mole-Skull-Classification-Expert
