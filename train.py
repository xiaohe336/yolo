# coding:utf-8
from ultralytics import YOLO

# 数据集配置文件路径
data_yaml_path = '/home/user/PycharmProjects/yolo/Cigarettesbutts/data.yaml'  # 确保这个路径是正确的，并且data.yaml文件格式正确

# 初始化YOLO模型，加载预训练权重
model = YOLO('yolo11n.pt')

# 训练模型
# 注意：这里的参数可能需要根据您的具体需求和数据集大小进行调整
if __name__ == '__main__':
    results = model.train(data=data_yaml_path,
                          epochs=300,          # 训练轮次
                          batch=32,          # 批处理大小
                          project='Cigarette',  # 项目名称，用于保存训练结果
                          name='v1Adam_n',          # 训练名称，用于区分不同的训练实验
                          optimizer='Adam',    # 使用Adam优化器
                          lr0=0.001,          # 初始学习率
                          weight_decay=5e-4,  # 权重衰减（L2正则化）
                          )
# 训练完成后，results对象将包含训练过程中的各种信息，如损失曲线、精度等
