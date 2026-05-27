# 贡献指南 | Contributing Guidelines

感谢您对本项目的关注！我们欢迎各种形式的贡献，包括但不限于代码提交、问题报告、功能建议、文档改进等。

## 📖 目录

- [行为准则](#-行为准则)
- [贡献方式](#-贡献方式)
- [开发环境设置](#-开发环境设置)
- [代码规范](#-代码规范)
- [提交流程](#-提交流程)
- [测试要求](#-测试要求)
- [代码审查](#-代码审查)

## 🤝 行为准则

### 我们的承诺

为了营造一个开放、友好的社区环境，我们承诺：

- 尊重不同观点和经验
- 接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 不可接受的行为

- 使用性化的语言或图像
- 人身攻击或侮辱性评论
- 公开或私下骚扰
- 未经许可发布他人隐私信息
- 其他不道德或不专业的行为

## 🎯 贡献方式

### 1. 报告问题

发现 bug 或有功能建议？请提交 Issue：

- **Bug 报告**：提供详细的复现步骤、环境信息、错误日志
- **功能建议**：描述使用场景、期望行为、可能的实现方案
- **文档改进**：指出不清楚或有误的文档部分

### 2. 提交代码

#### 适合新手的任务

查看标记为 [`good first issue`](https://github.com/yourusername/mole-classification-agent/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 的任务：

- 文档完善
- 代码注释补充
- 简单的 bug 修复
- 单元测试编写

#### 高级贡献

- 新功能开发
- 性能优化
- 架构改进
- 依赖升级

### 3. 分享经验

- 撰写使用教程
- 分享应用案例
- 在社交媒体推广项目
- 在学术会议展示成果

## 🛠️ 开发环境设置

### 1. Fork 与克隆

```bash
# 1. 在 GitHub 上 Fork 本仓库
# 2. 克隆您的 fork
git clone https://github.com/YOUR_USERNAME/mole-classification-agent.git
cd mole-classification-agent

# 3. 添加上游仓库
git remote add upstream https://github.com/yourusername/mole-classification-agent.git
```

### 2. 创建开发环境

```bash
# 创建 Conda 环境
conda create -n mole_agent_dev python=3.11
conda activate mole_agent_dev

# 安装开发依赖
pip install -r agent_system/requirements.txt

# 安装开发工具
pip install black flake8 pytest mypy
```

### 3. 配置 API Key

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

### 4. 验证安装

```bash
cd agent_system
streamlit run app.py
```

## 📝 代码规范

### Python 代码风格

遵循 [PEP 8](https://pep8.org/) 规范：

```python
# ✅ 好的命名
def classify_image(image_path):
    """对单个图像进行分类"""
    pass

# ❌ 避免的命名
def cls(img):  # 太简短
    pass
```

### 文档字符串

所有公共函数和类必须包含文档字符串：

```python
def upload_images(image_files: List[str]) -> str:
    """
    上传多个图像文件并返回存储目录
    
    Args:
        image_files: 图像文件路径列表
        
    Returns:
        存储目录的 UUID 字符串
        
    Raises:
        ValueError: 当文件不是有效的图像格式时
    """
    pass
```

### 类型注解

鼓励使用类型注解：

```python
from typing import Dict, List, Optional

def classify(
    image_dir: str,
    confidence_threshold: float = 0.5
) -> Dict[str, Optional[str]]:
    pass
```

### 代码格式化

使用 Black 进行代码格式化：

```bash
# 安装 Black
pip install black

# 格式化代码
black agent_system/*.py

# 检查代码风格
flake8 agent_system/*.py
```

## 🔄 提交流程

### 1. 创建分支

```bash
# 从主分支创建新分支
git checkout -b feature/your-feature-name

# 或修复 bug
git checkout -b fix/bug-fix-description
```

分支命名规范：
- `feature/xxx` - 新功能
- `fix/xxx` - Bug 修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 代码重构
- `test/xxx` - 测试相关

### 2. 进行更改

- 保持提交小而专注
- 每个提交解决一个问题
- 编写清晰的提交信息

### 3. 编写提交信息

```bash
# 好的提交信息
git commit -m "feat: 添加多语言支持

- 支持中文和英文界面
- 添加语言切换按钮
- 更新相关文档"

# 避免的提交信息
git commit -m "更新代码"  # 太模糊
git commit -m "修复 bug"  # 不具体
```

提交信息格式：
```
<type>(<scope>): <subject>

<body>

<footer>
```

类型包括：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

### 4. 推送分支

```bash
git push origin feature/your-feature-name
```

### 5. 创建 Pull Request

1. 在 GitHub 上导航到您的 fork
2. 点击 "Compare & pull request"
3. 填写 PR 描述：
   - 简要说明更改内容
   - 关联相关 Issue（如适用）
   - 添加截图（UI 更改时）
4. 等待代码审查

## ✅ 测试要求

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_image_classifier.py

# 生成覆盖率报告
pytest --cov=agent_system tests/
```

### 编写测试

新功能应包含相应的测试：

```python
import pytest
from image_classifier import ImageClassifierService

def test_classify_success():
    """测试分类成功的情况"""
    classifier = ImageClassifierService()
    result = classifier.classify("path/to/test/images")
    
    assert result["status"] == "success"
    assert result["genus"] is not None

def test_classify_invalid_images():
    """测试无效图像的处理"""
    classifier = ImageClassifierService()
    result = classifier.classify("path/to/invalid/images")
    
    assert result["status"] == "error"
```

### 测试覆盖率

目标覆盖率：
- 核心模块：>80%
- 服务层：>70%
- UI 层：>50%

## 🔍 代码审查

### 审查标准

代码审查将关注以下方面：

1. **功能正确性**
   - 代码是否按预期工作
   - 是否处理了边界情况
   - 错误处理是否完善

2. **代码质量**
   - 是否遵循代码规范
   - 代码是否简洁清晰
   - 是否有重复代码

3. **性能影响**
   - 是否有性能退化
   - 是否有内存泄漏风险
   - 是否有优化空间

4. **测试覆盖**
   - 是否包含足够的测试
   - 测试是否覆盖边界情况

5. **文档完整性**
   - 是否更新了相关文档
   - 代码注释是否清晰
   - API 变更是否说明

### 审查反馈

审查者会提供建设性反馈：

```markdown
### 需要修改
- [ ] 第 23 行：添加错误处理
- [ ] 第 45 行：变量命名不够清晰

### 建议改进
- [ ] 考虑使用列表推导式简化代码
- [ ] 可以添加性能日志

### 做得好的
- ✅ 测试覆盖很全面
- ✅ 文档字符串很清晰
```

### 回应审查

- 保持开放心态
- 积极讨论技术方案
- 及时回应审查意见
- 必要时更新代码

## 📚 学习资源

- [Python 最佳实践](https://docs.python-guide.org/)
- [Git 工作流](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [Pull Request 指南](https://docs.github.com/en/pull-requests)
- [代码审查艺术](https://google.github.io/eng-practices/)

## ❓ 常见问题

**Q: 我是新手，不知道从哪里开始？**

A: 查看标记为 "good first issue" 的任务，或直接在 Issue 中询问。

**Q: 我的代码被拒绝了怎么办？**

A: 不要气馁！审查反馈是学习的机会。根据建议修改后重新提交。

**Q: 我可以同时处理多个任务吗？**

A: 建议一次专注于一个任务，确保质量。

**Q: 如何联系维护者？**

A: 在 Issue 中留言，或通过邮件联系（见 README）。

## 🙏 致谢

感谢所有为本项目做出贡献的开发者！

---

<div align="center">

**期待您的贡献！一起打造更好的项目！**

[返回首页](README.md)

</div>
