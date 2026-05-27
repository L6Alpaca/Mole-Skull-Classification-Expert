import os
import sys
import torch
import numpy as np
import json
import time

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from tools.get_sample_predict import get_sample_predict
import torchvision


class ImageClassificationEvaluator:
    """纯图像分类评估器 - 验证分类准确率和混淆矩阵"""

    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_indict = None
        self.num_classes = 18
        self.class_names = []

    def load_model(self, weights_path, num_classes=18):
        """加载EfficientNet-B3模型"""
        self.num_classes = num_classes
        self.model = torchvision.models.efficientnet_b3()
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        print("模型加载完成: {}".format(weights_path))

    def load_labels(self, json_path):
        """加载类别标签映射"""
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)
        self.class_names = [self.class_indict[str(i)] for i in range(len(self.class_indict))]
        print("加载了 {} 个类别标签".format(len(self.class_names)))
        print("类别列表: {}".format(self.class_names))

    def predict_individual(self, ind_path):
        """预测单个个体"""
        predict_cla = get_sample_predict(ind_path, self.model, self.device, self.num_classes)
        return predict_cla

    def get_genus_from_filename(self, filename):
        """从文件名提取属名 - 文件名格式: Genus#species#..."""
        parts = filename.split('#')
        if len(parts) >= 1:
            return parts[0]
        return None

    def get_species_from_filename(self, filename):
        """从文件名提取物种名"""
        parts = filename.split('#')
        if len(parts) >= 2:
            return parts[1]
        return None

    def evaluate(self, test_path):
        """评估测试集"""
        y_true = []
        y_pred = []

        print("测试路径: {}".format(test_path))

        individual_dirs = []
        for root, dirs, files in os.walk(test_path):
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(dirs) == 0 and len(jpg_files) > 0:
                individual_dirs.append(root)

        print("找到 {} 个测试个体".format(len(individual_dirs)))

        correct = 0
        total = 0
        details = []

        start_time = time.time()

        for idx, ind_path in enumerate(individual_dirs):
            files = os.listdir(ind_path)
            jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if len(jpg_files) == 0:
                continue

            true_genus = self.get_genus_from_filename(jpg_files[0])

            if true_genus is None:
                print("警告: 无法从文件 {} 提取属名".format(jpg_files[0]))
                continue

            try:
                pred_label_idx = self.predict_individual(ind_path)
                pred_label = self.class_indict[str(pred_label_idx)]
            except Exception as e:
                print("预测失败 {}: {}".format(ind_path, e))
                continue

            y_true.append(true_genus)
            y_pred.append(pred_label)

            is_correct = true_genus == pred_label
            if is_correct:
                correct += 1
            total += 1

            details.append({
                'path': ind_path,
                'true': true_genus,
                'pred': pred_label,
                'correct': is_correct
            })

            if total % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / total
                remaining = avg_time * (len(individual_dirs) - total)
                print("进度: {}/{}, 当前准确率: {:.4f}, 预计剩余时间: {:.1f}秒".format(
                    total, len(individual_dirs), correct/total, remaining))

        total_time = time.time() - start_time

        print("\n评估完成: {} 个样本, 耗时: {:.2f}秒".format(total, total_time))
        if total > 0:
            print("平均推理时间: {:.3f}秒/样本".format(total_time/total))
            accuracy = correct / total
        else:
            accuracy = 0
        print("整体准确率: {:.4f} ({:.2f}%)".format(accuracy, accuracy*100))

        return y_true, y_pred, details, total_time

    def compute_confusion_matrix(self, y_true, y_pred):
        """手动计算混淆矩阵"""
        classes = sorted(list(set(y_true) | set(y_pred)))
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            if true in class_to_idx and pred in class_to_idx:
                cm[class_to_idx[true]][class_to_idx[pred]] += 1

        return cm, classes

    def print_classification_report(self, y_true, y_pred):
        """手动生成分类报告"""
        classes = sorted(list(set(y_true) | set(y_pred)))
        class_correct = {c: 0 for c in classes}
        class_total = {c: 0 for c in classes}

        for true, pred in zip(y_true, y_pred):
            class_total[true] += 1
            if true == pred:
                class_correct[true] += 1

        total_correct = sum(class_correct.values())
        total_samples = sum(class_total.values())
        overall_acc = total_correct / total_samples if total_samples > 0 else 0

        print("\n" + "=" * 70)
        print("纯图像分类评估报告")
        print("=" * 70)
        print("\n整体准确率: {:.4f} ({:.2f}%)".format(overall_acc, overall_acc*100))
        print("样本总数: {}\n".format(len(y_true)))

        print("-" * 70)
        print("{:<25} {:>10} {:>10} {:>10} {:>10}".format('类别', '精确率', '召回率', 'F1分数', '支持数'))
        print("-" * 70)

        for c in classes:
            total = class_total[c]
            correct = class_correct[c]
            precision = correct / total if total > 0 else 0
            recall = correct / total if total > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print("{:<25} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(c, precision, recall, f1, total))

        print("-" * 70)

        return {
            'classes': classes,
            'class_correct': class_correct,
            'class_total': class_total,
            'overall_accuracy': overall_acc
        }

    def print_confusion_matrix_text(self, cm, classes):
        """文本方式打印混淆矩阵"""
        print("\n混淆矩阵:")
        print("-" * 70)

        header = "{:<20}".format("真实/预测")
        for c in classes:
            short_c = c[:12] if len(c) > 12 else c
            header += "{:>12}".format(short_c)
        print(header)
        print("-" * 70)

        for i, c in enumerate(classes):
            short_c = c[:12] if len(c) > 12 else c
            row = "{:<20}".format(short_c)
            for j in range(len(classes)):
                row += "{:>12}".format(cm[i][j])
            print(row)
        print("-" * 70)


def main():
    project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
    weights_path = os.path.join(project_root, "weights/EfficientNet-B3/best_network.pth")
    label_path = os.path.join(project_root, "data/genus_labels.json")
    test_path = os.path.join(project_root, "data/test")

    print("=" * 70)
    print("纯图像分类准确率验证测试")
    print("=" * 70)
    print("项目根目录: {}".format(project_root))
    print("权重文件: {}".format(weights_path))
    print("标签文件: {}".format(label_path))
    print("测试路径: {}".format(test_path))

    if not os.path.exists(weights_path):
        print("错误: 权重文件不存在 - {}".format(weights_path))
        return
    if not os.path.exists(label_path):
        print("错误: 标签文件不存在 - {}".format(label_path))
        return
    if not os.path.exists(test_path):
        print("错误: 测试数据不存在 - {}".format(test_path))
        return

    evaluator = ImageClassificationEvaluator()
    evaluator.load_model(weights_path)
    evaluator.load_labels(label_path)

    print("\n开始评估...")
    y_true, y_pred, details, total_time = evaluator.evaluate(test_path)

    if len(y_true) == 0:
        print("错误: 没有收集到任何预测结果")
        return

    report = evaluator.print_classification_report(y_true, y_pred)

    cm, classes = evaluator.compute_confusion_matrix(y_true, y_pred)
    evaluator.print_confusion_matrix_text(cm, classes)

    report_path = os.path.join(project_root, "docs/image_only_classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("纯图像分类评估报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("整体准确率: {:.4f} ({:.2f}%)\n".format(report['overall_accuracy'], report['overall_accuracy']*100))
        f.write("样本总数: {}\n".format(len(y_true)))
        f.write("总耗时: {:.2f}秒\n".format(total_time))
        f.write("平均推理时间: {:.3f}秒/样本\n\n".format(total_time/len(y_true) if len(y_true) > 0 else 0))
        f.write("各类别统计:\n")
        for c in report['classes']:
            f.write("  {}: 正确={}, 总数={}\n".format(c, report['class_correct'][c], report['class_total'][c]))
        f.write("\n混淆矩阵:\n")
        f.write("类别: {}\n".format(classes))
        f.write(str(cm) + "\n\n")
        f.write("详细结果:\n")
        for d in details:
            status = "正确" if d['correct'] else "错误"
            f.write("  [{}] {}: 真实={}, 预测={}\n".format(status, os.path.basename(d['path']), d['true'], d['pred']))

    print("\n详细报告已保存: {}".format(report_path))

    inference_stats = {
        'total_samples': len(y_true),
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(y_true) if len(y_true) > 0 else 0,
        'accuracy': report['overall_accuracy']
    }

    print("\n" + "=" * 70)
    print("推理性能指标汇总")
    print("=" * 70)
    print("总样本数: {}".format(inference_stats['total_samples']))
    print("总耗时: {:.2f}秒".format(inference_stats['total_time']))
    print("平均推理时间: {:.3f}秒/样本".format(inference_stats['avg_time_per_sample']))
    print("整体准确率: {:.4f} ({:.2f}%)".format(inference_stats['accuracy'], inference_stats['accuracy']*100))
    print("=" * 70)


if __name__ == "__main__":
    main()