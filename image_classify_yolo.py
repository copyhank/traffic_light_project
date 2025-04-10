import torch
import cv2
import numpy as np
from ultralytics import YOLO

def classify_image(image_path, model_path):
    # 讀取 YOLO 模型
    model = YOLO(model_path)
    
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image.")
        return
    
    # 進行推論
    results = model(image)
    
    # 解析結果
    ArrowSequence = ["Left", "Up, Stright", "Right","Left/","Up/", "Right/"] #表示三種class的方向
    for result in results:
        if hasattr(result, 'probs') and result.probs is not None:
            # 取得分類機率最大者
            predicted_class = result.probs.top1
            confidence = result.probs.data[predicted_class].item()
            print(f"Predicted Class: {ArrowSequence[predicted_class]}, Confidence: {confidence:.2f}")
        else:
            print("No classification results found.")

# 測試程式
image_path = r"output/splited_green_2.jpg"  # 替換成你的測試圖片路徑
#model_path = "bestSingleArrowE9.pt"  # 替換成你的訓練後模型權重路徑
#model_path = "bestSingleArrowE40.pt"  # 替換成你的訓練後模型權重路徑
model_path = "bestSingleArrowTrain17_6class.pt"  # 替換成你的訓練後模型權重路徑
classify_image(image_path, model_path)