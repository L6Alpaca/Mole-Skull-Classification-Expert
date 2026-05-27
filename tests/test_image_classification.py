import unittest
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from tools.get_sample_predict import get_sample_predict


class TestImageClassification(unittest.TestCase):
    """图像分类单元测试"""
    
    def setUp(self):
        """测试前准备"""
        self.device = torch.device("cpu")
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # 创建测试图像
        self.create_test_images()
    
    def create_test_images(self):
        """创建测试用的假图像"""
        # 创建一个模拟个体文件夹
        test_individual_dir = os.path.join(self.test_data_dir, "test_individual")
        os.makedirs(test_individual_dir, exist_ok=True)
        
        # 创建四个视角的测试图像
        views = ["s#l", "s#d", "s#v", "m#l"]
        for view in views:
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
            img_path = os.path.join(test_individual_dir, f"test#spec#{view}.JPG")
            img.save(img_path)
        
        self.test_individual_dir = test_individual_dir
    
    def test_get_sample_predict_basic(self):
        """测试get_sample_predict基本功能"""
        # 创建一个简单的测试模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
                self.fc = torch.nn.Linear(16 * 111 * 111, 5)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel().to(self.device)
        model.eval()
        
        result = get_sample_predict(self.test_individual_dir, model, self.device, 5)
        
        # 验证返回类型和值范围
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 0)  # 标量值
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 5)
    
    def test_get_sample_predict_with_probability(self):
        """测试get_sample_predict返回概率分布"""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
                self.fc = torch.nn.Linear(16 * 111 * 111, 5)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel().to(self.device)
        model.eval()
        
        result = get_sample_predict(self.test_individual_dir, model, self.device, 5, is_genus=False)
        
        # 验证概率分布
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 5))
        # 验证概率和为1
        self.assertAlmostEqual(float(result.sum()), 1.0, places=5)
    
    def test_image_extension_filtering(self):
        """测试图像扩展名过滤"""
        # 在测试文件夹中添加非图像文件
        non_image_path = os.path.join(self.test_individual_dir, "not_an_image.txt")
        with open(non_image_path, 'w') as f:
            f.write("test")
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
                self.fc = torch.nn.Linear(16 * 111 * 111, 5)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel().to(self.device)
        model.eval()
        
        # 应该正常工作，忽略txt文件
        result = get_sample_predict(self.test_individual_dir, model, self.device, 5)
        self.assertIsInstance(result, np.ndarray)
        
        # 清理
        os.remove(non_image_path)
    
    def test_empty_directory(self):
        """测试空目录处理"""
        empty_dir = os.path.join(self.test_data_dir, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3)
                self.fc = torch.nn.Linear(16 * 111 * 111, 5)
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = SimpleModel().to(self.device)
        model.eval()
        
        # 空目录应该返回0（默认分类）
        result = get_sample_predict(empty_dir, model, self.device, 5)
        self.assertEqual(result, 0)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)


if __name__ == '__main__':
    unittest.main()