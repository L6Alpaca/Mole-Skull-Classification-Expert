# 鼹鼠分类智能体 - 快速开始指南

## 环境要求
- Python 3.8+
- conda 环境（推荐使用 ai 环境）

## 安装步骤

### 1. 激活 conda 环境
```powershell
conda activate ai
```

### 2. 安装 PyTorch (CPU版本)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装其他依赖
```powershell
pip install streamlit langchain langchain-community langchain-chroma langchain-core chromadb dashscope numpy Pillow python-multipart
```

或者直接运行安装脚本：
```powershell
cd agent_system
install_deps.bat
```

### 4. 设置 API Key
```powershell
$env:DASHSCOPE_API_KEY = "你的API密钥"
```

## 测试系统

### 运行环境检查
```powershell
python quick_check.py
```

### 运行简化测试
```powershell
python test_simple.py
```

## 启动应用

### 启动 Web 界面
```powershell
streamlit run app.py
```

然后在浏览器中打开显示的地址（通常是 http://localhost:8501）

## 使用说明

### 1. 上传图像
- 在左侧边栏点击上传按钮
- 选择4个不同视角的鼹鼠图像
- 支持格式：jpg, jpeg, png

### 2. 进行分类
- 点击"开始识别"按钮
- 等待分类完成

### 3. 智能问答
- 在主界面输入问题
- 系统会结合分类结果和知识库回答

## 常见问题

### ModuleNotFoundError: No module named 'torch'
- 确保已激活正确的 conda 环境
- 运行 PyTorch 安装命令

### ModuleNotFoundError: No module named 'XXX'
- 运行完整的依赖安装命令
- 或者使用 install_deps.bat

### 分类不准确
- 确保上传4个完整视角的图像
- 使用高质量、清晰的图像

### API Key 问题
- 确保设置了 DASHSCOPE_API_KEY 环境变量
- 检查 API Key 是否有效
