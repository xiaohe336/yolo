# -*- coding: utf-8 -*-
"""
数据集分割脚本 - 适配Labelme转YOLO格式
将button10文件夹下的Labelme格式数据分割为训练集和验证集
"""

import os
import random
import shutil
import json
from sklearn.model_selection import train_test_split

def split_dataset(input_folder='button10', output_folder='button10_Dataset', train_ratio=0.8, random_seed=42):
    """
    分割数据集为训练集和验证集
    
    Args:
        input_folder: 输入数据文件夹路径
        output_folder: 输出数据集文件夹路径
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在")
        return
    
    # 检查images和labels文件夹
    images_dir = os.path.join(input_folder, 'images')
    labels_dir = os.path.join(input_folder, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"错误: '{images_dir}' 文件夹不存在")
        return
    
    if not os.path.exists(labels_dir):
        print(f"错误: '{labels_dir}' 文件夹不存在")
        return
    
    # 获取所有图像文件（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("错误: 在images文件夹中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 获取对应的JSON标注文件
    valid_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        json_file = base_name + '.json'
        json_path = os.path.join(labels_dir, json_file)
        
        if os.path.exists(json_path):
            # 验证JSON文件格式
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                # 检查是否包含必要的字段
                if 'imagePath' in json_data and 'shapes' in json_data:
                    valid_files.append((img_file, json_file))
                else:
                    print(f"警告: JSON文件 {json_file} 格式不正确")
            except Exception as e:
                print(f"警告: 无法读取JSON文件 {json_file}: {e}")
        else:
            print(f"警告: 图像 {img_file} 没有对应的JSON标注文件")
    
    print(f"有效的图像-标注对: {len(valid_files)} 个")
    
    if not valid_files:
        print("错误: 没有找到有效的图像-标注对")
        return
    
    # 分割训练集和验证集
    train_files, val_files = train_test_split(
        valid_files, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 创建输出目录结构
    output_dirs = [
        os.path.join(output_folder, 'images', 'train'),
        os.path.join(output_folder, 'images', 'val'),
        os.path.join(output_folder, 'labelme_jsons', 'train'),
        os.path.join(output_folder, 'labelme_jsons', 'val')
    ]
    
    for dir_path in output_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 复制文件到相应目录
    def copy_files(file_list, split_type):
        """复制文件到指定分割类型目录"""
        for img_file, json_file in file_list:
            # 复制图像文件
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_folder, 'images', split_type, img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制JSON标注文件
            src_json = os.path.join(labels_dir, json_file)
            dst_json = os.path.join(output_folder, 'labelme_jsons', split_type, json_file)
            shutil.copy2(src_json, dst_json)
        
        print(f"复制完成: {split_type}集 - {len(file_list)} 个文件")
    
    # 复制训练集和验证集文件
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    
    # 创建labels目录结构（为YOLO格式输出准备）
    labels_dirs = [
        os.path.join(output_folder, 'labels', 'train'),
        os.path.join(output_folder, 'labels', 'val')
    ]
    
    for dir_path in labels_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    print(f"\n数据集分割完成!")
    print(f"输出目录结构:")
    print_dataset_structure(output_folder)
    
    # 生成数据集信息报告
    generate_dataset_report(output_folder, train_files, val_files)

def generate_dataset_report(output_folder, train_files, val_files):
    """生成数据集统计报告"""
    print("\n" + "="*50)
    print("数据集统计报告")
    print("="*50)
    
    # 统计训练集信息
    train_shapes_count = 0
    train_keypoints_count = 0
    
    for img_file, json_file in train_files:
        json_path = os.path.join(output_folder, 'labelme_jsons', 'train', json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                train_shapes_count += len(data.get('shapes', []))
                # 统计关键点数量
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') == 'point':
                        train_keypoints_count += 1
        except:
            pass
    
    # 统计验证集信息
    val_shapes_count = 0
    val_keypoints_count = 0
    
    for img_file, json_file in val_files:
        json_path = os.path.join(output_folder, 'labelme_jsons', 'val', json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                val_shapes_count += len(data.get('shapes', []))
                # 统计关键点数量
                for shape in data.get('shapes', []):
                    if shape.get('shape_type') == 'point':
                        val_keypoints_count += 1
        except:
            pass
    
    print(f"训练集:")
    print(f"  - 图像数量: {len(train_files)}")
    print(f"  - 标注形状总数: {train_shapes_count}")
    print(f"  - 关键点总数: {train_keypoints_count}")
    print(f"  - 平均每图像标注数: {train_shapes_count/len(train_files):.2f}")
    
    print(f"验证集:")
    print(f"  - 图像数量: {len(val_files)}")
    print(f"  - 标注形状总数: {val_shapes_count}")
    print(f"  - 关键点总数: {val_keypoints_count}")
    print(f"  - 平均每图像标注数: {val_shapes_count/len(val_files):.2f}")
    
    print(f"总计:")
    print(f"  - 总图像数量: {len(train_files) + len(val_files)}")
    print(f"  - 总标注形状数: {train_shapes_count + val_shapes_count}")
    print(f"  - 总关键点数: {train_keypoints_count + val_keypoints_count}")

def print_dataset_structure(folder_path):
    """打印数据集目录结构"""
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return
    
    print(f"📁 {folder_path}/")
    for root, dirs, files in os.walk(folder_path):
        level = root.replace(folder_path, '').count(os.sep)
        indent = '  ' * level
        if level == 0:
            continue
        
        # 只显示到第二级目录
        if level <= 2:
            dir_name = os.path.basename(root)
            if level == 1:
                print(f"{indent}├─📁 {dir_name}/")
            else:
                print(f"{indent}└─📁 {dir_name}/")
            
            # 显示前5个文件（如果存在）
            if files and level == 2:
                file_indent = '  ' * (level + 1)
                for i, file in enumerate(files[:5]):
                    if i == len(files[:5]) - 1 and len(files) <= 5:
                        print(f"{file_indent}└─ {file}")
                    else:
                        print(f"{file_indent}├─ {file}")
                if len(files) > 5:
                    print(f"{file_indent}└─ ... 和其他 {len(files) - 5} 个文件")

def check_input_structure(folder_path='button10'):
    """检查输入文件夹结构"""
    print("检查输入文件夹结构...")
    
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹 '{folder_path}' 不存在")
        return False
    
    images_dir = os.path.join(folder_path, 'images')
    labels_dir = os.path.join(folder_path, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ '{images_dir}' 文件夹不存在")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"❌ '{labels_dir}' 文件夹不存在")
        return False
    
    # 统计文件数量
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    json_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
    
    print(f"✅ 图像文件: {len(image_files)} 个")
    print(f"✅ JSON标注文件: {len(json_files)} 个")
    
    # 检查匹配的文件对
    matched_pairs = 0
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        json_file = base_name + '.json'
        if json_file in json_files:
            matched_pairs += 1
    
    print(f"✅ 匹配的图像-标注对: {matched_pairs} 个")
    
    return len(image_files) > 0 and len(json_files) > 0

def main():
    """主函数"""
    print("=" * 50)
    print("Labelme数据集分割脚本")
    print("=" * 50)
    
    # 检查输入结构
    if not check_input_structure('button10'):
        print("\n❌ 输入文件夹结构不正确，请检查!")
        return
    
    print("\n开始分割数据集...")
    
    # 分割数据集
    split_dataset(
        input_folder='button10',
        output_folder='button10_Dataset',
        train_ratio=0.8,  # 80%训练集，20%验证集
        random_seed=42
    )
    
    print("\n🎉 Labelme数据集分割完成!")
    print("现在你可以运行 Labelme转YOLO-Keypoint 脚本来转换标注格式了")

if __name__ == "__main__":
    main()