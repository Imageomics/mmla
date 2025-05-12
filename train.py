from ultralytics import YOLO

model = YOLO("yolo11m.pt")
results = model.train(
    data="data/dataset.yaml",
    epochs=50,
    imgsz=640,
    device=[0,1]
)