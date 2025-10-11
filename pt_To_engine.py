from ultralytics import YOLO

# Load a model
model = YOLO("/home/user/PycharmProjects/yolo/button/v1Adam_n2/weights/best.pt")  # load an official model
# Export the model
model.export(format="engine")
