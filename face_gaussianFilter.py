from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import torch
# import fiftyone as fo
# import fiftyone.zoo as foz
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ["Human head"]
label_field = "ground_truth"
if __name__ == "__main__":
    # print(os.getcwd())
    """
    # try:
    #     dataset = foz.load_zoo_dataset(
    #         "open-images-v7",
    #         label_types=["detections"],
    #         classes=classes,
    #           splits=[""]
    #         max_samples=1000,  # Adjust this number based on your needs
    #         dataset_dir="/content/open-images-v7"
    #     )
    #     dataset.export(
    #         export_dir="/content/open-images-v7-yolo",
    #         dataset_type=fo.types.YOLOv5Dataset,
    #         label_field=label_field,
    #         classes=classes,
    #     )
    #
    # except Exception as e:
    #     print(e)
    """
    model = YOLO("runs/detect/train5/weights/best.pt")
    model.to(device)

    im1 = Image.open("00a3ee845477b6ca.jpg")
    cap = cv2.VideoCapture('Planning Poker Is A LIE! [ciaGj27dcIk].mp4')
    cap.get(cv2.CAP_PROP_FPS)
    results = model(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, cap.get(cv2.CAP_PROP_FPS), results[0].orig_shape[1], results[0].orig_shape[0])
    # results = model.train(data="/content\open-images-v7-yolo/dataset.yaml", epochs=8, imgsz=640)

