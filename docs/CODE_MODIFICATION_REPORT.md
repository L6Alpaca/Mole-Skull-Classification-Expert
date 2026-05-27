# 鼹鼠头骨四视角图像分类代码重构报告

## 1. 项目概述

本项目原本基于四视角图像和头骨尺寸数据进行后融合分类。本次重构的目标是提取并优化仅使用图像数据进行分类识别的代码模块，移除所有尺寸相关的预处理、特征提取、模型输入及融合分类逻辑。

## 2. 移除的尺寸相关代码模块

### 2.1 主脚本中的尺寸相关代码

在 `predict_ind_ToSpecies copy.py` 中移除的代码模块：

| 移除内容 | 行号 | 说明 |
|---------|------|------|
| `SizeInfoLoader` 类定义 | 20-77 | 完整的尺寸信息加载器类，包含从Excel读取尺寸数据和提取尺寸特征向量的功能 |
| `size_loader = SizeInfoLoader()` | 88 | 尺寸加载器实例化 |
| `size_model_path` 变量 | 117 | 尺寸模型文件路径定义 |
| `size_features` 获取 | 142 | 获取个体尺寸特征 |
| `size_features` 和 `size_model_path` 参数传递 | 145 | 传递尺寸信息到预测函数 |

### 2.2 工具函数中的尺寸相关代码

在 `tools/get_sample_predict_species.py` 中移除的代码模块：

| 移除内容 | 说明 |
|---------|------|
| `from tools.get_size_prob import predict_size_prob` | 尺寸概率预测导入 |
| `import tools.get_size_prob as sizepart` | 尺寸模块导入 |
| 融合分类逻辑 | 尺寸与图像概率融合的相关代码 |

### 2.3 尺寸处理模块

保留但标记为独立模块（不参与纯图像分类流程）：

| 文件 | 状态 | 说明 |
|------|------|------|
| `tools/get_size_prob.py` | 保留 | 完整的尺寸处理模块，包含尺寸模型加载、特征工程和融合分类函数 |
| `size_part/` 目录 | 保留 | 尺寸模型训练和测试的完整代码目录 |

### 2.4 数据文件

| 文件/目录 | 状态 | 说明 |
|-----------|------|------|
| `data/test/*/size.xlsx` | 保留 | 个体尺寸数据Excel文件 |
| `data/Euroscaptor.xlsx` | 保留 | 尺寸数据表 |

## 3. 保留的图像分类核心代码

### 3.1 核心图像预测函数

**文件**: `tools/get_sample_predict.py`

**功能说明**:
- 遍历个体文件夹中的所有图像
- 根据视角类型（s#l, s#d, s#v, m#l）进行加权融合
- 支持返回分类结果（is_genus=True）或概率分布（is_genus=False）

**核心逻辑**:
1. 图像预处理：ToTensor + Normalize
2. 单图像推理：使用EfficientNet-B3模型提取特征并输出概率
3. 视角加权融合：根据各视角准确率计算权重，进行加权平均
4. 结果输出：返回类别索引或概率分布

### 3.2 物种分类器

**文件**: `tools/SpeciesClassfier_ind.py`

**功能说明**:
- 加载7个属级分类模型（Euroscaptor, Mogera, Parascaptor, Scapanus, Scaptonyx, Talpa, Uropsilus）
- 根据属级预测结果选择对应的物种分类模型进行二级分类

### 3.3 文件工具函数

**文件**: `tools/file_utils.py`

**功能说明**:
- 提供文件操作、文本分割等辅助功能
- `text_segmentation` 函数用于解析图像文件名中的信息

### 3.4 GPU检测工具

**文件**: `tools/GPU_Detecter.py`

**功能说明**:
- 检测可用的GPU设备

## 4. 新增文件

### 4.1 纯图像分类入口

**文件**: `predict_image_only.py`

**功能说明**:
- 独立的纯图像分类入口
- 移除了所有尺寸相关代码
- 包含 `ImageOnlyClassifier` 类，提供：
  - `load_genus_model()`: 加载属级分类模型
  - `load_label_mapping()`: 加载类别标签映射
  - `predict_single_individual()`: 预测单个个体
  - `batch_predict()`: 批量预测

### 4.2 单元测试

**文件**: `tests/test_image_classification.py`

**测试用例**:
| 测试方法 | 测试内容 |
|---------|---------|
| `test_get_sample_predict_basic` | 测试基本预测功能 |
| `test_get_sample_predict_with_probability` | 测试概率分布输出 |
| `test_image_extension_filtering` | 测试图像扩展名过滤 |
| `test_empty_directory` | 测试空目录处理 |

### 4.3 集成测试

**文件**: `tests/test_integration.py`

**测试用例**:
| 测试方法 | 测试内容 |
|---------|---------|
| `test_classifier_initialization` | 测试分类器初始化 |
| `test_model_loading` | 测试模型加载 |
| `test_single_individual_prediction` | 测试单个个体预测 |
| `test_batch_prediction` | 测试批量预测 |
| `test_inference_performance` | 测试推理性能 |

## 5. 测试结果分析

### 5.1 测试环境

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows 10 |
| Python版本 | 3.11.7 |
| PyTorch版本 | 2.x |
| 设备 | CPU |

### 5.2 单元测试结果

```
测试文件: tests/test_image_classification.py
测试用例: 4个
全部通过 ✓
```

### 5.3 集成测试结果

```
测试文件: tests/test_integration.py
测试用例: 5个
全部通过 ✓
```

### 5.4 推理性能测试

| 指标 | 结果 |
|------|------|
| 单次推理平均时间 | < 5秒 |
| 测试次数 | 5次 |
| 图像数量 | 4张（四视角） |

## 6. 代码结构变更对比

### 6.1 重构前结构

```
项目根目录/
├── predict_ind_ToSpecies copy.py    # 包含尺寸融合的主脚本
├── tools/
│   ├── get_sample_predict.py        # 图像预测
│   ├── get_sample_predict_species.py # 包含尺寸融合的预测
│   ├── get_size_prob.py             # 尺寸处理模块
│   ├── SpeciesClassfier_ind.py      # 物种分类器
│   ├── file_utils.py                # 文件工具
│   └── GPU_Detecter.py              # GPU检测
├── size_part/                       # 尺寸模型训练目录
└── data/
    └── test/                        # 测试数据（含size.xlsx）
```

### 6.2 重构后结构

```
项目根目录/
├── predict_ind_ToSpecies copy.py    # 原始脚本（保留）
├── predict_image_only.py            # 新增：纯图像分类入口
├── tools/
│   ├── get_sample_predict.py        # 纯图像预测（核心）
│   ├── get_sample_predict_species.py # 原始版本（保留）
│   ├── get_size_prob.py             # 尺寸模块（保留）
│   ├── SpeciesClassfier_ind.py      # 物种分类器
│   ├── file_utils.py                # 文件工具
│   └── GPU_Detecter.py              # GPU检测
├── size_part/                       # 尺寸模型训练目录（保留）
├── tests/                           # 新增：测试目录
│   ├── test_image_classification.py # 单元测试
│   └── test_integration.py          # 集成测试
└── data/
    └── test/                        # 测试数据
```

## 7. 使用说明

### 7.1 运行纯图像分类

```python
from predict_image_only import ImageOnlyClassifier

# 初始化分类器
classifier = ImageOnlyClassifier()

# 加载模型和标签
classifier.load_genus_model("weights/EfficientNet-B3/best_network.pth")
classifier.load_label_mapping("data/genus_labels.json")

# 预测单个个体
result = classifier.predict_single_individual("path/to/individual/folder")

# 批量预测
results = classifier.batch_predict("path/to/test/data")
```

### 7.2 运行测试

```bash
# 运行单元测试
python tests/test_image_classification.py

# 运行集成测试
python tests/test_integration.py

# 使用pytest（推荐）
python -m pytest tests/ -v
```

## 8. 总结

### 8.1 重构成果

1. **成功移除尺寸相关代码**：从主流程中完全分离了尺寸数据处理逻辑
2. **保留图像分类核心功能**：四视角加权融合逻辑完整保留
3. **创建独立入口**：`predict_image_only.py` 提供纯图像分类的简洁接口
4. **完善测试覆盖**：添加了单元测试和集成测试，验证功能正确性和性能

### 8.2 后续建议

1. 可考虑将模型权重路径和配置参数外部化到配置文件
2. 增加更多数据增强选项以提升模型鲁棒性
3. 考虑添加模型量化和优化以提升推理速度
4. 可扩展支持其他图像分类模型架构