"""
图像分类服务模块
实现分层识别：先识别属级别，再识别种级别
"""
import os
import json
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from tools.get_sample_predict import get_sample_predict
import config


class SpeciesClassifier:
    """
    种级别分类器类
    针对多型属进行种级别分类
    """
    
    def __init__(self, device=None):
        """
        初始化种级别分类器
        :param device: 设备，默认为自动选择
        """
        self.device = device if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # 加载物种标签
        self.species_labels = self._load_species_labels()
        
        # 初始化各属的种级别分类模型
        self.models = {}
        self._init_species_models()
    
    def _load_species_labels(self):
        """加载物种标签映射"""
        try:
            with open(config.SPECIES_LABELS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"物种标签加载失败: {e}")
            raise
    
    def _init_species_models(self):
        """初始化各个多型属的种级别分类模型"""
        for genus in config.POLYTYPIC_GENERA:
            num_classes = len(self.species_labels[genus])
            model = torchvision.models.efficientnet_b3()
            model.classifier[1] = torch.nn.Linear(
                model.classifier[1].in_features, num_classes
            )
            
            # 加载模型权重
            weights_path = os.path.join(
                config.SPECIES_CLASSIFIER_DIR,
                "weights",
                f"EB3_{genus}",
                "best_network.pth"
            )
            
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
                model = model.to(self.device)
                model.eval()
                self.models[genus] = model
                print(f"种级别分类器 {genus} 加载成功，共 {num_classes} 个物种")
            else:
                print(f"警告：种级别分类器权重文件不存在: {weights_path}")
                self.models[genus] = None
    
    def classify_species(self, genus_name, image_dir):
        """
        根据属名进行种级别分类
        :param genus_name: 属名
        :param image_dir: 包含四个视角图像的文件夹路径
        :return: 物种名称，如果无法分类则返回None
        """
        if genus_name not in config.POLYTYPIC_GENERA:
            return None
        
        model = self.models.get(genus_name)
        if model is None:
            return None
        
        try:
            num_classes = len(self.species_labels[genus_name])
            predict_class_idx = get_sample_predict(
                image_dir, model, self.device, num_classes
            )
            species_name = self.species_labels[genus_name][int(predict_class_idx)]
            return species_name
        except Exception as e:
            print(f"种级别分类失败: {e}")
            return None


class ImageClassifierService:
    """
    图像分类服务类
    提供分层识别功能：先属级别，再种级别
    """

    def __init__(self, device=None):
        """
        初始化图像分类服务
        :param device: 设备，默认为自动选择
        """
        self.device = device if device else torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.genus_model = None
        self.genus_class_indict = None
        self.species_classifier = None
        
        self._load_genus_model()
        self._load_genus_labels()
        self._init_species_classifier()

    def _load_genus_model(self):
        """加载属级分类模型"""
        try:
            print(f"正在加载属级别分类模型: {config.GENUS_MODEL_PATH}")
            self.genus_model = torchvision.models.efficientnet_b3()
            self.genus_model.classifier[1] = torch.nn.Linear(
                self.genus_model.classifier[1].in_features, 18
            )
            self.genus_model = self.genus_model.to(self.device)
            self.genus_model.load_state_dict(
                torch.load(config.GENUS_MODEL_PATH, map_location="cpu")
            )
            self.genus_model.eval()
            print("属级别分类模型加载成功!")
        except Exception as e:
            print(f"属级别分类模型加载失败: {e}")
            raise

    def _load_genus_labels(self):
        """加载属级标签映射"""
        try:
            with open(config.GENUS_LABELS_PATH, "r", encoding="utf-8") as f:
                self.genus_class_indict = json.load(f)
            print(f"属级别标签加载成功，共 {len(self.genus_class_indict)} 个属")
        except Exception as e:
            print(f"属级别标签加载失败: {e}")
            raise
    
    def _init_species_classifier(self):
        """初始化种级别分类器"""
        try:
            print("正在初始化种级别分类器...")
            self.species_classifier = SpeciesClassifier(self.device)
            print("种级别分类器初始化成功!")
        except Exception as e:
            print(f"种级别分类器初始化失败: {e}")
            self.species_classifier = None

    def classify(self, image_dir):
        """
        对单个个体的四个视角图像进行分层分类
        :param image_dir: 包含四个视角图像的文件夹路径
        :return: 分类结果字典，包含属名、种名、置信度等信息
        """
        if self.genus_model is None or self.genus_class_indict is None:
            raise RuntimeError("服务未正确初始化，请检查模型和标签文件")

        try:
            # 第一步：属级别分类
            predict_class_idx = get_sample_predict(
                image_dir, self.genus_model, self.device, 18
            )
            genus_name = self.genus_class_indict[str(predict_class_idx)]
            
            # 第二步：种级别分类（仅对多型属）
            species_name = None
            if genus_name in config.POLYTYPIC_GENERA and self.species_classifier:
                species_name = self.species_classifier.classify_species(genus_name, image_dir)
            
            # 如果无法进行种级别分类，返回属名作为结果
            final_name = species_name if species_name else genus_name
            
            result = {
                "genus": genus_name,
                "species": species_name,
                "final_name": final_name,
                "class_idx": predict_class_idx,
                "status": "success",
                "message": "分类成功"
            }
            return result
        except Exception as e:
            return {
                "genus": None,
                "species": None,
                "final_name": None,
                "class_idx": None,
                "status": "error",
                "message": f"分类失败: {str(e)}"
            }

    def get_class_names(self):
        """获取所有属类别名称"""
        if self.genus_class_indict:
            return [self.genus_class_indict[str(i)] for i in range(len(self.genus_class_indict))]
        return []


if __name__ == "__main__":
    # 简单测试
    print("测试图像分类服务...")
    service = ImageClassifierService()
    
    # 测试获取类别名称
    print(f"所有属类别: {service.get_class_names()}")
    
    # 测试分类（使用测试数据）
    test_dir = str(Path(__file__).parent.parent / "data" / "test" / "Euroscaptor sp1" / "Euroscaptor sp1_HK1412001")
    if os.path.exists(test_dir):
        result = service.classify(test_dir)
        print(f"分类结果: {result}")