from ultralytics import YOLO

# Load a model
model = YOLO("/home/user/PycharmProjects/yolo/Cigarette/v1Adam_n/weights/best.pt")  # load an official model
# Export the model
model.export(format="engine")
