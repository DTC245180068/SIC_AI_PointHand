# -*- coding: utf-8 -*-
from ultralytics import YOLO
from train_cnn import build_and_train_cnn

if __name__ == "__main__":
    build_and_train_cnn(epochs=25)
    YOLO("yolov8n.pt").train(
        data="E:/Du_an_cu_chi_tay/Hand Gesture.v1i.yolov8/data.yaml",
        epochs=15, imgsz=320, batch=16, workers=2,
        optimizer="AdamW", lr0=0.002, cos_lr=True, amp=True, cache="ram", degrees=15.0,
    )
