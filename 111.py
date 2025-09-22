from ultralytics import YOLO

model = YOLO(r"D:\Cigarette_Switch_Detector\Cigarette\v13\weights\best.pt")

# 直接使用验证集进行INT8校准
model.export(
    format="engine",
    int8=True,
    data="data.yaml",        # 包含验证集路径的配置文件
    calib_subset="val",      # 使用验证集
    calib_num_images=100,    # 使用100张图像
    workspace=4
)