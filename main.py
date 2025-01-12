from ultralytics import YOLO
import os 


def main(): 
    epochs = 25
    batch_size = 16
    device = "0,1"

    lr_start = 0.001
    lr_end = 0.0001

    version = 'v1.0'

    # Train
    want_train = input("Train (y/n)? ")
    if want_train == 'y':
        model = YOLO("yolo11n.pt")
        train_result = model.train(data = "data/data.yaml", epochs = epochs, 
                                   imgsz = 416, batch = 8, device = 0, cos_lr = True, lr0 = 0.001, lrf = 0.0001, 
                                   project = 'train_result', name=version)
        return 1

    model = YOLO("runs/detect/" + version + "best.pt")
    metrics = model.val(data = "data/data.yaml", split = "test")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")
    print(f"F1-Score: {metrics.box.f1:.4f}")
    




if __name__ == '__main__':
    main()
