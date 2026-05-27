import os
import pandas as pd
from typing import List, Tuple

# 定义常见图片格式（可根据需求补充）
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',".JPG")

def get_deepest_subfolders(root_dir: str) -> List[str]:
    """获取根目录下所有最深层级的子文件夹（无下级子文件夹）"""
    deepest_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:  # 无子文件夹则为最深层级
            deepest_folders.append(dirpath)
    print(f"找到 {len(deepest_folders)} 个最深层级文件夹")
    return deepest_folders

def get_images_in_folder(folder_path: str) -> List[Tuple[str, str]]:
    """获取文件夹内所有图片的「完整路径」和「原始图片名称（含后缀）」"""
    images = []
    for filename in os.listdir(folder_path):
        if filename.startswith('.'):  # 忽略隐藏文件（如Mac的.DS_Store）
            continue
        if filename.endswith(IMAGE_EXTENSIONS):  # 仅保留指定格式图片（原始后缀）
            img_path = os.path.join(folder_path, filename)
            images.append((img_path, filename))  # 保留原始文件名（无标准化）
    print(f"  文件夹内有效图片数量：{len(images)} 张")
    return images

def process_folder_with_priority(root_dir: str, excel_path: str):
    """
    核心逻辑（优先级：文件夹名匹配「个体」列 > 图片名匹配「图片名称」列）
    1. 遍历最深文件夹，先尝试用文件夹名匹配Excel「个体」列
    2. 若匹配成功：保存该个体对应的所有行数据
    3. 若匹配失败：读取文件夹内所有图片，用图片名匹配「图片名称」列
    4. 生成汇总表和剩余数据
    """
    # 1. 读取Excel数据并预处理（仅去空值，保留原始格式）
    df = pd.read_excel(excel_path)
    required_cols = ["个体", "图片名称"]
    
    # 检查必要列是否存在
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Excel表中未找到「{col}」列，请检查列名是否正确")
    
    # 预处理：仅去除两列的空值行（不修改原始名称）
    df_clean = df.dropna(subset=required_cols).copy()
    print(f"\nExcel数据预处理完成：共 {len(df_clean)} 条有效数据（仅去空值）")
    
    # 2. 获取所有最深文件夹并处理
    deepest_folders = get_deepest_subfolders(root_dir)
    all_filtered_data = []  # 存储所有匹配数据（汇总用）
    matched_individuals = []  # 存储通过「个体」列匹配成功的个体名
    matched_img_names = []    # 存储通过「图片名称」列匹配成功的图片名
    
    for folder in deepest_folders:
        folder_name = os.path.basename(folder)  # 文件夹名（即待匹配的个体名）
        print(f"\n===== 处理文件夹：{folder_name} =====")
        matched_df = pd.DataFrame()  # 存储当前文件夹的匹配结果
        
        # --------------------------
        # 第一步：优先按文件夹名匹配「个体」列（精确匹配，无标准化）
        # --------------------------
        individual_matched_df = df_clean[df_clean["个体"] == folder_name].copy()
        if not individual_matched_df.empty:
            matched_df = individual_matched_df
            print(f"  ✅ 优先级1匹配成功：文件夹名「{folder_name}」与Excel「个体」列匹配")
            print(f"  匹配数据行数：{len(matched_df)} 条")
            # 记录匹配的个体名（用于后续剩余数据计算）
            matched_individuals.append(folder_name)
        else:
            print(f"  ❌ 优先级1匹配失败：文件夹名「{folder_name}」未在Excel「个体」列找到")
            # --------------------------
            # 第二步：匹配失败，按图片名匹配「图片名称」列（精确匹配）
            # --------------------------
            images = get_images_in_folder(folder)
            if not images:
                print("  ⚠️  无有效图片，跳过后续匹配")
                continue
            
            # 提取原始图片名称列表（含后缀）
            img_names = [img_name for _, img_name in images]
            # 精确匹配图片名称
            img_matched_df = df_clean[df_clean["图片名称"].isin(img_names)].copy()
            
            if not img_matched_df.empty:
                matched_df = img_matched_df
                print(f"  ✅ 优先级2匹配成功：{len(matched_df)} 张图片与Excel「图片名称」列匹配")
                # 记录匹配的图片名
                matched_img_names.extend(img_matched_df["图片名称"].tolist())
            else:
                print(f"  ❌ 优先级2匹配失败：文件夹内图片未在Excel「图片名称」列找到匹配")
        
        # --------------------------
        # 保存当前文件夹的匹配结果
        # --------------------------
        if not matched_df.empty:
            save_path = os.path.join(folder, "size.xlsx")
            matched_df.to_excel(save_path, index=False)
            print(f"  ✅ 匹配结果已保存到：{os.path.basename(save_path)}")
            all_filtered_data.append(matched_df)
    
    # --------------------------
    # 3. 生成根目录汇总表（所有匹配数据合并）
    # --------------------------
    if all_filtered_data:
        all_filtered_df = pd.concat(all_filtered_data, ignore_index=True).drop_duplicates()
        all_filtered_path = os.path.join(root_dir, "test.xlsx")
        all_filtered_df.to_excel(all_filtered_path, index=False)
        print(f"\n===== 汇总结果 =====")
        print(f"✅ 所有匹配数据已保存：{os.path.basename(all_filtered_path)}（{len(all_filtered_df)} 条）")
    else:
        print(f"\n===== 汇总结果 =====")
        print(f"⚠️  无任何匹配数据，未生成汇总表")
    
    # --------------------------
    # 4. 生成剩余数据（Excel中未被任何方式匹配的行）
    # --------------------------
    # 未被「个体」列匹配，且未被「图片名称」列匹配的行
    remaining_df = df_clean[
        (~df_clean["个体"].isin(matched_individuals)) & 
        (~df_clean["图片名称"].isin(matched_img_names))
    ].copy()
    remaining_path = os.path.join(root_dir, "last.xlsx")
    remaining_df.to_excel(remaining_path, index=False)
    print(f"✅ 未匹配数据已保存：{os.path.basename(remaining_path)}（{len(remaining_df)} 条）")

if __name__ == "__main__":
    try:
        # 输入路径（支持绝对路径/相对路径）
        root_folder = "data/test"
        excel_file = "data/Euroscaptor.xlsx"
        # 验证路径有效性
        if not os.path.isdir(root_folder):
            raise FileNotFoundError(f"根文件夹不存在：{root_folder}")
        if not os.path.isfile(excel_file):
            raise FileNotFoundError(f"Excel文件不存在：{excel_file}")
        
        # 执行处理
        print("===== 开始处理 =====")
        process_folder_with_priority(root_folder, excel_file)
        print("\n===== 所有处理完成 =====")
    
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")