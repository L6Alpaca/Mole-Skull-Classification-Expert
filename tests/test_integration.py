import unittest
import os
import sys
import torch
import torchvision
import numpy as np
import time
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from predict_image_only import ImageOnlyClassifier


class TestIntegration(unittest.TestCase):
    """图像分类集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device("cpu")
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "integration_test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 创建测试数据结构
        self.create_test_data_structure()
    
    def create_test_data_structure(self):
        """创建模拟的测试数据结构"""
        # 创建属类别文件夹
        genus_dir = os.path.join(self.test_data_dir, "test")
        os.makedirs(genus_dir, exist_ok=True)
        
        # 创建物种文件夹
        species_dir = os.path.join(genus_dir, "Euroscaptor_test")
        os.makedirs(species_dir, exist_ok=True)
        
        # 创建个体文件夹
        individual_dir = os.path.join(species_dir, "Euroscaptor_test_001")
        os.makedirs(individual_dir, exist_ok=True)
        
        # 创建四个视角的测试图像
        views = ["s#l", "s#d", "s#v", "m#l"]
        for view in views:
            img = Image.new('RGB', (224, 224), color=(128 + np.random.randint(0, 50), 
                                                       128 + np.random.randint(0, 50), 
                                                       128 + np.random.randint(0, 50)))
            img_path = os.path.join(individual_dir, f"Euroscaptor#test#001#{view}.JPG")
            img.save(img_path)
        
        self.test_individual_dir = individual_dir
        self.test_root_dir = genus_dir
    
    def test_classifier_initialization(self):
        """测试分类器初始化"""
        classifier = ImageOnlyClassifier(device=self.device)
        
        # 验证设备设置正确
        self.assertEqual(classifier.device, self.device)
        
        # 验证模型初始状态
        self.assertIsNone(classifier.model)
        
        # 验证transform存在
        self.assertIsNotNone(classifier.test_transform)
    
    def test_model_loading(self):
        """测试模型加载"""
        classifier = ImageOnlyClassifier(device=self.device)
        
        # 创建一个临时的dummy权重文件
        dummy_model = torchvision.models.efficientnet_b3()
        dummy_model.classifier[1] = torch.nn.Linear(dummy_model.classifier[1].in_features, 18)
        
        weights_path = os.path.join(self.test_data_dir, "dummy_weights.pth")
        torch.save(dummy_model.state_dict(), weights_path)
        
        # 测试加载模型
        classifier.load_genus_model(weights_path)
        
        # 验证模型已加载
        self.assertIsNotNone(classifier.model)
        self.assertTrue(classifier.model.training is False)  # 应该处于评估模式
    
    def test_single_individual_prediction(self):
        """测试单个个体预测"""
        import torchvision
        
        classifier = ImageOnlyClassifier(device=self.device)
        
        # 创建并保存dummy模型
        dummy_model = torchvision.models.efficientnet_b3()
        dummy_model.classifier[1] = torch.nn.Linear(dummy_model.classifier[1].in_features, 18)
        dummy_model.eval()
        
        weights_path = os.path.join(self.test_data_dir, "dummy_weights.pth")
        torch.save(dummy_model.state_dict(), weights_path)
        
        classifier.load_genus_model(weights_path)
        
        # 创建测试图像
        test_img_dir = os.path.join(self.test_data_dir, "single_test")
        os.makedirs(test_img_dir, exist_ok=True)
        
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        img.save(os.path.join(test_img_dir, "test#spec#s#l.JPG"))
        
        # 测试预测
        result = classifier.predict_single_individual(test_img_dir)
        
        # 验证结果
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 0)
    
    def test_batch_prediction(self):
        """测试批量预测"""
        import torchvision
        
        classifier = ImageOnlyClassifier(device=self.device)
        
        # 创建并保存dummy模型
        dummy_model = torchvision.models.efficientnet_b3()
        dummy_model.classifier[1] = torch.nn.Linear(dummy_model.classifier[1].in_features, 2)
        dummy_model.eval()
        
        weights_path = os.path.join(self.test_data_dir, "dummy_weights.pth")
        torch.save(dummy_model.state_dict(), weights_path)
        
        # 创建label映射文件
        label_path = os.path.join(self.test_data_dir, "labels.json")
        import json
        with open(label_path, 'w') as f:
            json.dump({"0": "GenusA", "1": "GenusB"}, f)
        
        classifier.load_genus_model(weights_path, num_classes=2)
        classifier.load_label_mapping(label_path)
        
        # 创建测试数据结构
        batch_test_dir = os.path.join(self.test_data_dir, "batch_test", "data")
        os.makedirs(os.path.join(batch_test_dir, "GenusA", "Ind1"), exist_ok=True)
        os.makedirs(os.path.join(batch_test_dir, "GenusB", "Ind2"), exist_ok=True)
        
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        img.save(os.path.join(batch_test_dir, "GenusA", "Ind1", "test#A#s#l.JPG"))
        img.save(os.path.join(batch_test_dir, "GenusB", "Ind2", "test#B#s#l.JPG"))
        
        # 测试批量预测
        result = classifier.batch_predict(batch_test_dir)
        
        # 验证结果结构
        self.assertIsInstance(result, dict)
    
    def test_inference_performance(self):
        """测试推理性能"""
        import torchvision
        
        classifier = ImageOnlyClassifier(device=self.device)
        
        # 创建dummy模型
        dummy_model = torchvision.models.efficientnet_b3()
        dummy_model.classifier[1] = torch.nn.Linear(dummy_model.classifier[1].in_features, 18)
        dummy_model.eval()
        
        weights_path = os.path.join(self.test_data_dir, "dummy_weights.pth")
        torch.save(dummy_model.state_dict(), weights_path)
        
        classifier.load_genus_model(weights_path)
        
        # 多次推理计算平均时间
        num_runs = 5
        total_time = 0
        
        for _ in range(num_runs):
            start_time = time.time()
            classifier.predict_single_individual(self.test_individual_dir)
            total_time += time.time() - start_time
        
        avg_time = total_time / num_runs
        
        # 验证推理时间合理（小于5秒）
        self.assertLess(avg_time, 5.0)
        print(f"平均推理时间: {avg_time:.4f}秒")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)


if __name__ == '__main__':
    unittest.main()