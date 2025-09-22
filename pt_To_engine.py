from ultralytics import YOLO

# Load a model
model = YOLO(r"D:\Cigarette_Switch_Detector\Cigarette\v1Adam\weights\best.pt")  # load an official model
# Export the model
model.export(format="engine")
