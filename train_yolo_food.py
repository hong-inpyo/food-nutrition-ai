from ultralytics import YOLO
from multiprocessing import freeze_support

def main():

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="foodseg103.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )

if __name__ == "__main__":
    freeze_support()
    main()