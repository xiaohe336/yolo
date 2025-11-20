# -*- coding: utf-8 -*-
"""
Labelme转YOLO-Keypoint-批量

同济子豪兄 2023-4-17
"""

# 导入工具包
import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# 删除系统自动生成的多余文件

def clean_system_files():
    """删除系统自动生成的多余文件"""
    
    # 查看待删除的多余文件
    print("查找__MACOSX文件...")
    os.system('find . -iname "__MACOSX"')
    
    print("查找.DS_Store文件...")
    os.system('find . -iname ".DS_Store"')
    
    print("查找.ipynb_checkpoints文件...")
    os.system('find . -iname ".ipynb_checkpoints"')
    
    # 删除多余文件
    print("删除__MACOSX文件...")
    os.system('for i in `find . -iname "__MACOSX"`; do rm -rf $i;done')
    
    print("删除.DS_Store文件...")
    os.system('for i in `find . -iname ".DS_Store"`; do rm -rf $i;done')
    
    print("删除.ipynb_checkpoints文件...")
    os.system('for i in `find . -iname ".ipynb_checkpoints"`; do rm -rf $i;done')
    
    # 验证多余文件已删除
    print("验证__MACOSX文件已删除...")
    os.system('find . -iname "__MACOSX"')
    
    print("验证.DS_Store文件已删除...")
    os.system('find . -iname ".DS_Store"')
    
    print("验证.ipynb_checkpoints文件已删除...")
    os.system('find . -iname ".ipynb_checkpoints"')

# 数据集类别信息
Dataset_root = 'button10_Dataset'

# 框的类别
bbox_class = {
    'button':0  
}

# 关键点的类别
keypoint_class = ['1', '2', '3', '4']

def setup_directories():
    """设置目录结构"""
    os.chdir(Dataset_root)
    
    # 创建labels目录
    os.makedirs('labels/train', exist_ok=True)
    os.makedirs('labels/val', exist_ok=True)

# 函数-处理单个labelme标注json文件
def process_single_json(labelme_path, save_folder='../../labels/train'):
    """
    处理单个labelme标注json文件，转换为YOLO格式
    
    Args:
        labelme_path: labelme json文件路径
        save_folder: 保存YOLO格式txt文件的目录
    """
    
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)

    img_width = labelme['imageWidth']   # 图像宽度
    img_height = labelme['imageHeight'] # 图像高度

    # 生成 YOLO 格式的 txt 文件
    suffix = labelme_path.split('.')[-2]
    yolo_txt_path = suffix + '.txt'

    with open(yolo_txt_path, 'w', encoding='utf-8') as f:

        for each_ann in labelme['shapes']: # 遍历每个标注

            if each_ann['shape_type'] == 'rectangle': # 每个框，在 txt 里写一行

                yolo_str = ''

                ## 框的信息
                # 框的类别 ID
                bbox_class_id = bbox_class[each_ann['label']]
                yolo_str += '{} '.format(bbox_class_id)
                # 左上角和右下角的 XY 像素坐标
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                # 框中心点的 XY 像素坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                # 框宽度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                # 框高度
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                # 框中心点归一化坐标
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                # 框归一化宽度
                bbox_width_norm = bbox_width / img_width
                # 框归一化高度
                bbox_height_norm = bbox_height / img_height

                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)

                ## 找到该框中所有关键点，存在字典 bbox_keypoints_dict 中
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']: # 遍历所有标注
                    if each_ann['shape_type'] == 'point': # 筛选出关键点标注
                        # 关键点XY坐标、类别
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        if (x>bbox_top_left_x) & (x<bbox_bottom_right_x) & (y<bbox_bottom_right_y) & (y>bbox_top_left_y): # 筛选出在该个体框中的关键点
                            bbox_keypoints_dict[label] = [x, y]

                ## 把关键点按顺序排好
                for each_class in keypoint_class: # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2) # 2-可见不遮挡 1-遮挡 0-没有点
                    else: # 不存在的点，一律为0
                        yolo_str += '0 0 0 '
                # 写入 txt 文件中
                f.write(yolo_str + '\n')
                
    shutil.move(yolo_txt_path, save_folder)
    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))

def convert_train_labels():
    """转换训练集标注文件至`labels/train`目录"""
    os.chdir('labelme_jsons/train')
    
    save_folder = '../../labels/train'
    for labelme_path in os.listdir():
        try:
            process_single_json(labelme_path, save_folder=save_folder)
        except Exception as e:
            print('******有误******', labelme_path)
            print('错误信息:', e)
    print('YOLO格式的txt标注文件已保存至 ', save_folder)
    
    os.chdir('../../')

def convert_val_labels():
    """转换测试集标注文件至`labels/val`目录"""
    os.chdir('labelme_jsons/val')
    
    save_folder = '../../labels/val'
    for labelme_path in os.listdir():
        try:
            process_single_json(labelme_path, save_folder=save_folder)
        except Exception as e:
            print('******有误******', labelme_path)
            print('错误信息:', e)
    print('YOLO格式的txt标注文件已保存至 ', save_folder)
    
    os.chdir('../../')

def remove_labelme_jsons():
    """删除labelme格式的标注文件"""
    os.system('rm -rf labelme_jsons')

def show_directory_structure():
    """显示数据集目录结构"""
    try:
        import seedir as sd
        sd.seedir('button', style='emoji', depthlimit=2)
    except ImportError:
        print("请安装seedir和emoji包来显示目录结构")
        print("pip install seedir emoji")

def main():
    """主函数"""
    print("开始Labelme转YOLO-Keypoint批量转换...")
    
    # 清理系统文件
    clean_system_files()
    
    # 设置目录
    setup_directories()
    
    # 转换训练集
    convert_train_labels()
    
    # 转换验证集
    convert_val_labels()
    
    # 删除原始labelme文件
    remove_labelme_jsons()
    
    # 再次清理系统文件
    clean_system_files()
    
    # 显示最终目录结构
    print("最终数据集目录结构:")
    show_directory_structure()
    
    print("转换完成!")

if __name__ == "__main__":
    main()