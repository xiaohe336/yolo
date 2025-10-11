from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = "button/v1Adam_n2/weights/best.engine"
    test_path = "/home/user/PycharmProjects/yolo/yolo_dataset/val/images/1_frame_000076.jpg"

    # Load a custom model
    model = YOLO(pth_path)

    # Optionally load a custom configuration file if you have one
    # model.update(**{'hyp': 'path/to/your/custom_hyp.yaml'})

    # Predict with the model, setting confidence and IoU thresholds
    conf_threshold = 0.7  # Set the confidence threshold
    iou_threshold = 0.45  # Set the IoU threshold for NMS
    results = model(test_path, conf=conf_threshold, iou=iou_threshold, save=True)

    # Display results or further process them as needed
    # results.show()  # Uncomment to display results in a GUI window