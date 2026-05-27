import pandas as pd
import numpy as np
import pickle
import torch
from torchvision import transforms
from PIL import Image
import os
from math import exp
import math

# ===================== 尺寸分类核心函数（复用训练好的尺寸模型） =====================
def load_size_model(model_path):
    """加载训练好的尺寸分类模型组件"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"尺寸模型文件不存在：{model_path}")
    with open(model_path, "rb") as f:
        model_components = pickle.load(f)
    return {
        "model": model_components["best_model"],
        "scaler": model_components["scaler"],
        "label_encoder": model_components["label_encoder"],
        "feature_selector": model_components["feature_selector"],
        "global_view_means": model_components["global_view_means"],
        "global_stats": model_components["global_stats"],
        "view_order": model_components["view_order"],
        "all_feature_columns": model_components["all_feature_columns"],
        "class_names": model_components["label_encoder"].classes_
    }

def extract_view_type(image_name):
    """从文件名提取视角类型（与尺寸模型一致）"""
    if pd.isna(image_name):
        return None
    if "#m#l" in str(image_name):
        return "#m#l"
    elif "#s#l" in str(image_name):
        return "#s#l"
    elif "#s#v" in str(image_name):
        return "#s#v"
    elif "#s#d" in str(image_name):
        return "#s#d"
    else:
        return None

def enhanced_feature_engineering(df, global_stats=None):
    """尺寸特征工程（与训练逻辑一致）"""
    df_enhanced = df.copy()
    required_cols = ["length", "width"]
    for col in required_cols:
        if col not in df_enhanced.columns:
            raise ValueError(f"尺寸数据缺少必要列：{col}")
    
    df_enhanced["length"] = pd.to_numeric(df_enhanced["length"], errors="coerce").fillna(0)
    df_enhanced["width"] = pd.to_numeric(df_enhanced["width"], errors="coerce").fillna(0)
    
    # 基础特征（避免除0）
    width_safe = np.where(df_enhanced["width"] == 0, 1e-6, df_enhanced["width"])
    df_enhanced["aspect_ratio"] = df_enhanced["length"] / width_safe
    df_enhanced["area"] = df_enhanced["length"] * df_enhanced["width"]
    
    # 衍生特征（与训练时完全一致）
    df_enhanced["perimeter"] = 2 * np.pi * np.sqrt((df_enhanced["length"]**2 + df_enhanced["width"]**2) / 8)
    df_enhanced["size_diff"] = df_enhanced["length"] - df_enhanced["width"]
    length_safe = np.where(df_enhanced["length"] == 0, 1e-6, df_enhanced["length"])
    df_enhanced["inv_aspect_ratio"] = df_enhanced["width"] / length_safe
    perimeter_safe = np.where(df_enhanced["perimeter"] == 0, 1e-6, df_enhanced["perimeter"])
    df_enhanced["area_perimeter_ratio"] = df_enhanced["area"] / perimeter_safe
    area_safe = np.where(df_enhanced["area"] == 0, 1e-6, df_enhanced["area"])
    df_enhanced["length_square_over_area"] = (df_enhanced["length"]**2) / area_safe
    sum_length_width = df_enhanced["length"] + df_enhanced["width"]
    sum_safe = np.where(sum_length_width == 0, 1e-6, sum_length_width)
    df_enhanced["width_ratio"] = df_enhanced["width"] / sum_safe
    df_enhanced["log_length"] = np.log1p(df_enhanced["length"])
    df_enhanced["log_width"] = np.log1p(df_enhanced["width"])
    df_enhanced["log_area"] = np.log1p(df_enhanced["area"])
    df_enhanced["size_ratio"] = df_enhanced["length"] / sum_safe
    denominator = 4 * np.pi * area_safe
    denominator_safe = np.where(denominator == 0, 1e-6, denominator)
    df_enhanced["shape_complexity"] = (df_enhanced["perimeter"]**2) / denominator_safe
    
    # 标准化（基于尺寸模型的全局统计量）
    if global_stats is not None:
        length_mean = global_stats["length_mean"]
        length_std = global_stats["length_std"]
        width_mean = global_stats["width_mean"]
        width_std = global_stats["width_std"]
    else:
        length_mean = df_enhanced["length"].mean()
        length_std = df_enhanced["length"].std()
        width_mean = df_enhanced["width"].mean()
        width_std = df_enhanced["width"].std()
    
    length_std_safe = 1 if length_std == 0 else length_std
    width_std_safe = 1 if width_std == 0 else width_std
    df_enhanced["normalized_length"] = (df_enhanced["length"] - length_mean) / length_std_safe
    df_enhanced["normalized_width"] = (df_enhanced["width"] - width_mean) / width_std_safe
    
    return df_enhanced

def preprocess_size_data(size_df, size_model_config):
    """预处理尺寸数据（适配尺寸模型）"""
    global_view_means = size_model_config["global_view_means"]
    global_stats = size_model_config["global_stats"]
    view_order = size_model_config["view_order"]
    all_feature_columns = size_model_config["all_feature_columns"]
    raw_features = ['length', 'width']
    
    # 按个体分组（尺寸数据需包含“个体”列）
    if "个体" not in size_df.columns:
        raise ValueError("尺寸数据必须包含“个体”列")
    if "图片名称" not in size_df.columns:
        raise ValueError("尺寸数据必须包含“图片名称”列（用于匹配视角）")
    
    grouped = size_df.groupby("个体", as_index=False,sort = False)
    
    def process_individual(group):
        individual = group["个体"].iloc[0]
        expanded_raw = []
        
        # 填充每个视角的原始尺寸特征
        for view in view_order:
            view_data = group[group["图片名称"].apply(lambda x: extract_view_type(x) == view)]
            for col in raw_features:
                if len(view_data) > 0:
                    val = pd.to_numeric(view_data[col].iloc[0], errors="coerce") if col in view_data.columns else 0
                    expanded_raw.append(val if not pd.isna(val) else 0)
                else:
                    # 缺失视角用尺寸模型的全局均值填充
                    fill_val = global_view_means[view][col] if (global_view_means and view in global_view_means) else 0
                    expanded_raw.append(fill_val)
        
        # 推算衍生特征
        raw_feature_names = [f"{col}_{view}" for view in view_order for col in raw_features]
        raw_df = pd.DataFrame([expanded_raw], columns=raw_feature_names)
        
        view_dfs = []
        for view in view_order:
            view_raw_df = pd.DataFrame({
                "length": raw_df[f"length_{view}"],
                "width": raw_df[f"width_{view}"]
            })
            view_enhanced = enhanced_feature_engineering(view_raw_df, global_stats=global_stats)
            for col in view_enhanced.columns:
                view_enhanced.rename(columns={col: f"{col}_{view}"}, inplace=True)
            view_dfs.append(view_enhanced)
        view_combined = pd.concat(view_dfs, axis=1)
        view_combined["个体"] = individual
        return view_combined.iloc[0]
    
    expanded_df = grouped.apply(process_individual).reset_index(drop=True)
    test_features = expanded_df.drop("个体", axis=1)[all_feature_columns]
    return test_features, expanded_df["个体"].values

def predict_size_prob(individual_size_df, size_model_config):
    """预测单个个体的尺寸分类概率"""
    # 预处理当前个体的尺寸数据
    size_features, individuals = preprocess_size_data(individual_size_df, size_model_config)
    if len(individuals) != 1:
        raise ValueError("一次仅支持预测单个个体的尺寸概率")
    
    # 尺寸特征标准化和选择
    scaler = size_model_config["scaler"]
    feature_selector = size_model_config["feature_selector"]
    size_features_scaled = scaler.transform(size_features)
    size_features_selected = feature_selector.transform(size_features_scaled)
    
    # 预测概率
    size_model = size_model_config["model"]
    try:
        size_prob = size_model.predict_proba(size_features_selected)[0]  # 单个个体的概率分布
    except AttributeError:
        raise RuntimeError("尺寸模型不支持输出概率，请重新训练支持probability的模型")
    
    return size_prob

# ===================== 图像四视角分类函数（保留原有逻辑） =====================
def get_image_predict(sample_img_dir, model, device, class_num):
    """图像四视角加权分类（返回概率分布而非硬分类结果）"""
    # 图像各视角准确率（原有配置）
    s_l_acc = 0.9589
    s_d_acc = 0.9710
    s_v_acc = 0.9728
    m_d_acc = 0.9503

    # 图像视角权重计算（原有逻辑）
    current_dir = os.path.abspath(__file__)
    weight_cache_dir = os.path.join(os.path.dirname(current_dir), "__pycache__")
    sensitivity = 10
    if "weight_cache.npy" not in os.listdir(weight_cache_dir):
        ACC_MAX = max(s_l_acc, s_d_acc, s_v_acc, m_d_acc)
        s_l_weight = 1 - (ACC_MAX - s_l_acc) * sensitivity
        s_d_weight = 1 - (ACC_MAX - s_d_acc) * sensitivity
        s_v_weight = 1 - (ACC_MAX - s_v_acc) * sensitivity
        m_d_weight = 1 - (ACC_MAX - m_d_acc) * sensitivity
        weight_list = [s_d_weight, s_l_weight, s_v_weight, m_d_weight]
        np.save(os.path.join(weight_cache_dir, "weight_cache.npy"), weight_list)
    else:
        weight_list = np.load(os.path.join(weight_cache_dir, "weight_cache.npy"))
        s_d_weight = weight_list[0]
        s_l_weight = weight_list[1]
        s_v_weight = weight_list[2]
        m_d_weight = weight_list[3]

    # 图像预处理（原有逻辑）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 遍历当前个体的所有图像
    images = os.listdir(sample_img_dir)
    output = []
    for image in images:
        image_path = os.path.join(sample_img_dir, image)
        img = Image.open(image_path)
        img = transform(img).unsqueeze(dim=0)  # 增加batch维度
        with torch.no_grad():
            output_pre = torch.squeeze(model(img.to(device))).cpu()
            output_pre = torch.softmax(output_pre, dim=0)  # 单张图像的概率分布
        output.append(output_pre)

    # 图像视角加权融合（原有逻辑，返回概率分布而非硬分类）
    weighted_image_prob = torch.zeros(class_num)
    for i, image in enumerate(images):
        image_name = os.path.splitext(image)[0]
        image_flag = image_name.split("#")[4] + "#" + image_name.split("#")[5] if len(image_name.split("#")) > 5 else "p"
        if image_flag == "p":
            weighted_image_prob += 0.5 * output[i]
        elif image_flag == "s#l":
            weighted_image_prob += s_l_weight * output[i]
        elif image_flag == "s#d":
            weighted_image_prob += s_d_weight * output[i]
        elif image_flag == "s#v":
            weighted_image_prob += s_v_weight * output[i]
        elif image_flag == "m#l":
            weighted_image_prob += m_d_weight * output[i]

    # 归一化（确保概率和为1）
    weighted_image_prob = torch.softmax(weighted_image_prob, dim=0).numpy()
    return weighted_image_prob

# ===================== 尺寸+图像 加权融合分类（核心函数） =====================
def size_image_fusion_predict(
    individual_img_dir,  # 个体图像文件夹路径
    individual_size_df,  # 个体尺寸数据DataFrame（含“个体”“图片名称”“length”“width”）
    image_model,         # 图像分类模型
    size_model_config,   # 尺寸模型配置（load_size_model返回的字典）
    device,              # 计算设备（cuda/cpu）
    w_img=0.7,           # 图像特征权重（默认0.7，尺寸权重0.3）
    w_size=0.3           # 尺寸特征权重（默认0.3，需满足w_img + w_size = 1）
):
    """
    尺寸+图像加权融合分类
    返回：最终分类结果（类别索引）、各类别融合概率
    """
    # 校验权重合理性
    if not (0 <= w_img <= 1 and 0 <= w_size <= 1 and abs(w_img + w_size - 1) < 1e-6):
        raise ValueError("权重配置错误：w_img + w_size 必须等于1，且均在[0,1]范围内")
    
    # 1. 获取图像四视角加权概率
    class_num = len(size_model_config["class_names"])
    image_prob = get_image_predict(individual_img_dir, image_model, device, class_num)
    
    # 2. 获取尺寸分类概率
    size_prob = predict_size_prob(individual_size_df, size_model_config)
    
    # 3. 二次加权融合（概率级融合）
    fusion_prob = w_img * image_prob + w_size * size_prob
    fusion_prob = fusion_prob / np.sum(fusion_prob)  # 归一化（避免数值误差）
    
    # 4. 选择融合概率最高的类别
    final_pred = np.argmax(fusion_prob)
    
    return final_pred, fusion_prob

# ===================== 主函数（示例：批量处理多个个体） =====================
def batch_fusion_predict(
    root_dir,            # 根目录：每个子文件夹对应一个个体（含图像+尺寸excel）
    image_model,         # 图像模型
    size_model_path,     # 尺寸模型文件路径
    device,              # 计算设备
    w_img=0.7,           # 图像权重
    w_size=0.3           # 尺寸权重
):
    """
    批量处理多个个体的融合分类
    根目录结构：
    root_dir/
    ├─ 个体1/
    │  ├─ 图像1.jpg
    │  ├─ 图像2.jpg
    │  └─ 尺寸数据.xlsx（或与图像同名的excel，需包含“个体”“图片名称”“length”“width”）
    ├─ 个体2/
    │  └─ ...
    """
    # 加载尺寸模型
    size_model_config = load_size_model(size_model_path)
    class_names = size_model_config["class_names"]
    results = []
    
    # 遍历每个个体文件夹
    individual_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    for idx, individual in enumerate(individual_folders):
        print(f"正在处理个体 {idx+1}/{len(individual_folders)}: {individual}")
        individual_dir = os.path.join(root_dir, individual)
        
        # 1. 读取当前个体的尺寸数据（假设excel文件名为“尺寸数据.xlsx”，可根据实际修改）
        size_excel_path = os.path.join(individual_dir, "尺寸数据.xlsx")
        if not os.path.exists(size_excel_path):
            print(f"警告：个体 {individual} 缺少尺寸数据，跳过")
            continue
        size_df = pd.read_excel(size_excel_path)
        # 确保尺寸数据仅包含当前个体（避免多个体混杂）
        size_df = size_df[size_df["个体"] == individual].reset_index(drop=True)
        
        # 2. 图像文件夹路径（当前个体文件夹）
        img_dir = individual_dir
        
        # 3. 融合分类
        try:
            final_pred, fusion_prob = size_image_fusion_predict(
                individual_img_dir=img_dir,
                individual_size_df=size_df,
                image_model=image_model,
                size_model_config=size_model_config,
                device=device,
                w_img=w_img,
                w_size=w_size
            )
            
            # 整理结果
            result = {
                "个体": individual,
                "最终预测类别": class_names[final_pred],
                "最终预测类别索引": int(final_pred),
                "融合概率之和": round(np.sum(fusion_prob), 4)
            }
            # 添加每个类别的融合概率
            for cls_name, prob in zip(class_names, fusion_prob):
                result[f"融合概率_{cls_name}"] = round(prob, 4)
            # 添加图像/尺寸单独的概率（可选，用于分析）
            image_prob = get_image_predict(img_dir, image_model, device, len(class_names))
            size_prob = predict_size_prob(size_df, size_model_config)
            for cls_name, img_p, size_p in zip(class_names, image_prob, size_prob):
                result[f"图像概率_{cls_name}"] = round(img_p, 4)
                result[f"尺寸概率_{cls_name}"] = round(size_p, 4)
            
            results.append(result)
        except Exception as e:
            print(f"个体 {individual} 处理失败：{str(e)}")
            continue
    
    # 保存结果到Excel
    result_df = pd.DataFrame(results)
    output_path = os.path.join(root_dir, f"融合分类结果_图像权重{w_img}_尺寸权重{w_size}.xlsx")
    result_df.to_excel(output_path, index=False)
    print(f"\n批量处理完成！结果保存至：{output_path}")
    return result_df