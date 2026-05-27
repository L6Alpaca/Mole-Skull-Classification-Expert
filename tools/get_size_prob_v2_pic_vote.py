import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def predict_species_probability(size_df, model_components_path):
    """
    使用训练好的模型预测物种分类概率
    """
    
    # 1. 加载模型组件
    if not os.path.exists(model_components_path):
        raise FileNotFoundError(f"模型文件不存在: {model_components_path}")
    
    with open(model_components_path, 'rb') as f:
        model_components = pickle.load(f)
    
    best_model = model_components['best_model']
    scaler = model_components['scaler']
    le = model_components['label_encoder']
    feature_selector = model_components['feature_selector']
    selected_features = model_components['selected_features']
    all_feature_columns = model_components['all_feature_columns']
    global_stats = model_components['global_stats']
    fusion_method = model_components['fusion_method']
    
    print(f"加载模型成功: {type(best_model).__name__}")
    print(f"物种类别: {le.classes_}")
    print(f"训练时特征列: {all_feature_columns}")
    
    # 2. 使用与训练时完全相同的预处理函数
    def extract_view_type(image_name):
        """从图片名称中提取视角类型"""
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
        """与训练时完全相同的特征工程"""
        df_enhanced = df.copy()
        
        # 检查必要列
        required_cols = ["length", "width"]
        for col in required_cols:
            if col not in df_enhanced.columns:
                raise ValueError(f"数据中缺少必要列：{col}")
        
        # 确保数值类型
        df_enhanced["length"] = pd.to_numeric(df_enhanced["length"], errors="coerce").fillna(0)
        df_enhanced["width"] = pd.to_numeric(df_enhanced["width"], errors="coerce").fillna(0)
        
        # 基础特征
        width_safe = np.where(df_enhanced["width"] == 0, 1e-6, df_enhanced["width"])
        df_enhanced["aspect_ratio"] = df_enhanced["length"] / width_safe
        df_enhanced["area"] = df_enhanced["length"] * df_enhanced["width"]
        
        # 新增特征
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
        
        # 对数变换
        df_enhanced["log_length"] = np.log1p(df_enhanced["length"])
        df_enhanced["log_width"] = np.log1p(df_enhanced["width"])
        df_enhanced["log_area"] = np.log1p(df_enhanced["area"])
        
        df_enhanced["size_ratio"] = df_enhanced["length"] / sum_safe
        
        # 形状复杂性指标
        denominator = 4 * np.pi * area_safe
        denominator_safe = np.where(denominator == 0, 1e-6, denominator)
        df_enhanced["shape_complexity"] = (df_enhanced["perimeter"]**2) / denominator_safe
        
        # 标准化尺寸
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

    # 3. 预处理新数据
    print("正在进行数据预处理...")
    
    # 检查必要列
    required_cols = ["个体", "length", "width", "图片名称"]
    for col in required_cols:
        if col not in size_df.columns:
            raise ValueError(f"输入数据中缺少必要列：{col}")
    
    # 特征工程
    df_processed = enhanced_feature_engineering(size_df, global_stats)
    
    # 提取视角类型
    df_processed["view_type"] = df_processed["图片名称"].apply(extract_view_type)
    
    # 添加个体标识符
    df_processed["individual_id"] = df_processed["个体"]
    
    # 选择特征列（与训练时完全一致）
    feature_columns = ['length', 'width', 'aspect_ratio', 'area', 'perimeter', 
                      'size_diff', 'inv_aspect_ratio', 'area_perimeter_ratio', 
                      'length_square_over_area', 'width_ratio', 'log_length', 
                      'log_width', 'log_area', 'size_ratio', 'shape_complexity', 
                      'normalized_length', 'normalized_width']
    
    # 添加视角类型作为特征（独热编码）- 严格按照训练时的顺序
    view_dummies = pd.get_dummies(df_processed["view_type"], prefix="view")
    df_processed = pd.concat([df_processed, view_dummies], axis=1)
    
    # 确保所有视角列都存在
    expected_view_columns = ['view_#m#l', 'view_#s#l', 'view_#s#v', 'view_#s#d']
    for view_col in expected_view_columns:
        if view_col not in df_processed.columns:
            df_processed[view_col] = 0
    
    # 最终特征列（严格按照训练时的顺序）
    all_feature_columns_ordered = feature_columns + expected_view_columns
    
    # 检查特征是否匹配
    if set(all_feature_columns_ordered) != set(all_feature_columns):
        print(f"警告：特征不匹配！")
        print(f"当前特征: {sorted(all_feature_columns_ordered)}")
        print(f"训练特征: {sorted(all_feature_columns)}")
        
        # 使用训练时的特征顺序
        all_feature_columns_ordered = all_feature_columns
    
    # 确保所有特征都存在
    missing_features = set(all_feature_columns_ordered) - set(df_processed.columns)
    if missing_features:
        print(f"添加缺失特征: {missing_features}")
        for col in missing_features:
            df_processed[col] = 0
    
    # 构建最终数据集，严格按照训练时的特征顺序
    result_df = df_processed[["individual_id"] + all_feature_columns_ordered].copy()
    
    # 填充可能的缺失值
    result_df[all_feature_columns_ordered] = result_df[all_feature_columns_ordered].fillna(0)
    
    print(f"数据预处理完成：{len(result_df)}个样本")
    print(f"特征数量：{len(all_feature_columns_ordered)}")
    print(f"特征顺序：{all_feature_columns_ordered}")
    
    # 4. 特征标准化和选择
    X_new = result_df[all_feature_columns_ordered]
    individual_ids = result_df["individual_id"]
    
    print(f"标准化前数据形状: {X_new.shape}")
    print(f"特征名: {X_new.columns.tolist()}")
    
    # 关键修正：创建与训练时特征名匹配的DataFrame
    # 确保特征名称和顺序与训练时完全一致
    X_new_aligned = pd.DataFrame(X_new.values, columns=all_feature_columns)
    
    # 标准化
    X_new_scaled = scaler.transform(X_new_aligned)
    
    # 特征选择
    X_new_selected = feature_selector.transform(X_new_scaled)
    
    print(f"特征处理完成：{X_new_selected.shape[1]}个特征")
    
    # 5. 预测概率
    print("正在进行预测...")
    
    # 样本级别预测概率
    if hasattr(best_model, 'predict_proba'):
        sample_probabilities = best_model.predict_proba(X_new_selected)
    else:
        sample_predictions = best_model.predict(X_new_selected)
        n_classes = len(le.classes_)
        sample_probabilities = np.eye(n_classes)[sample_predictions]
    
    # 样本级别预测结果
    sample_predictions = np.argmax(sample_probabilities, axis=1)
    
    # 6. 个体级别预测融合
    def individual_level_prediction_predict(probabilities, individual_ids, fusion_method):
        """预测时的个体级别融合"""
        unique_individuals = np.unique(individual_ids)
        
        individual_predictions = []
        individual_probabilities = []
        
        for individual in unique_individuals:
            individual_indices = np.where(individual_ids == individual)[0]
            
            if fusion_method == 'probability_avg':
                individual_probs = np.mean(probabilities[individual_indices], axis=0)
                individual_pred = np.argmax(individual_probs)
            elif fusion_method == 'weighted_vote':
                weights = np.max(probabilities[individual_indices], axis=1)
                individual_probs = np.average(probabilities[individual_indices], 
                                            axis=0, weights=weights)
                individual_pred = np.argmax(individual_probs)
            else:
                individual_preds = np.argmax(probabilities[individual_indices], axis=1)
                individual_pred = np.argmax(np.bincount(individual_preds))
                individual_probs = np.mean(probabilities[individual_indices], axis=0)
            
            individual_probabilities.append(individual_probs)
            individual_predictions.append(individual_pred)
        
        return np.array(individual_predictions), np.array(individual_probabilities), unique_individuals
    
    # 应用个体级别融合
    individual_preds, individual_probs, unique_individuals = individual_level_prediction_predict(
        sample_probabilities, individual_ids.values, fusion_method
    )
    
    # 7. 整理结果
    species_names = le.classes_
    
    # 样本级别结果
    sample_results = []
    for i, (individual_id, pred, probs) in enumerate(zip(individual_ids, sample_predictions, sample_probabilities)):
        sample_result = {
            '样本索引': i,
            '个体ID': individual_id,
            '预测物种': species_names[pred],
            '预测物种编码': pred,
            '预测置信度': np.max(probs)
        }
        for j, species in enumerate(species_names):
            sample_result[f'{species}_概率'] = probs[j]
        sample_results.append(sample_result)
    
    sample_predictions_df = pd.DataFrame(sample_results)
    
    # 个体级别结果
    individual_results = []
    for i, (individual_id, pred, probs) in enumerate(zip(unique_individuals, individual_preds, individual_probs)):
        individual_result = {
            '个体ID': individual_id,
            '预测物种': species_names[pred],
            '预测物种编码': pred,
            '样本数量': np.sum(individual_ids == individual_id),
            '预测置信度': np.max(probs)
        }
        for j, species in enumerate(species_names):
            individual_result[f'{species}_概率'] = probs[j]
        individual_results.append(individual_result)
    
    individual_predictions_df = pd.DataFrame(individual_results)
    individual_predictions_df = individual_predictions_df.sort_values('预测置信度', ascending=False)
    
    print(f"预测完成：{len(unique_individuals)}个个体，{len(sample_predictions_df)}个样本")
    
    return individual_predictions_df, sample_predictions_df