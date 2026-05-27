import os
import sys

print("Python path:")
for p in sys.path:
    print("  ", p)

project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
print("\nProject root:", project_root)

weights_path = os.path.join(project_root, "weights/EfficientNet-B3/best_network.pth")
label_path = os.path.join(project_root, "data/genus_labels.json")
test_path = os.path.join(project_root, "data/test")

print("\nChecking files:")
print("  weights:", weights_path, "exists:", os.path.exists(weights_path))
print("  labels:", label_path, "exists:", os.path.exists(label_path))
print("  test:", test_path, "exists:", os.path.exists(test_path))

sys.path.append(project_root)

print("\nTrying to import tools.get_sample_predict...")
try:
    from tools.get_sample_predict import get_sample_predict
    print("  SUCCESS")
except Exception as e:
    print("  FAILED:", e)

print("\nTrying to import torchvision...")
try:
    import torchvision
    print("  SUCCESS, version:", torchvision.__version__)
except Exception as e:
    print("  FAILED:", e)

print("\nDone.")