# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测核心模块 (The "Chef")

该脚本的核心是 `process_img` 函数，它作为项目的核心API，
接收一张图片路径，并返回图中检测到的物体的结构化数据。

它被设计为可以被其他程序（如 run_cli.py 或外部测试程序）作为库导入和调用。
同时，当它被直接执行时 (python src/process.py)，它会运行一个独立的、
带自动保存功能的测试流程。

主要依赖: ultralytics, opencv-python, numpy, json, pathlib.
"""

import os
import time
import cv2
from ultralytics import YOLO
import numpy as np
import json
from pathlib import Path


# ---------------------------------------------------------------------------- #
#          默认配置 (主要用于 process_img 函数的直接调用)          #
# ---------------------------------------------------------------------------- #
# 这些全局变量作为 process_img 的默认配置。
# 它们可以被 configure_processor 函数动态修改，以实现灵活的配置。
DEFAULT_MODEL_WEIGHTS_PATH = "tennis_ball_runs/first_train2/weights/best.pt"  # 默认模型权重
DEFAULT_PROCESS_IMG_CONF_THRESHOLD = 0.20  # 默认置信度阈值
DEFAULT_PROCESS_IMG_IOU_THRESHOLD = 0.45   # 默认IOU阈值
DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME = "tennis_ball" # 默认目标类别

# ---------------------------------------------------------------------------- #
#          内部状态变量 (用于模型缓存)          #
# ---------------------------------------------------------------------------- #
# 以下全局变量用于在内存中缓存已加载的YOLO模型，避免在连续调用中反复从磁盘加载模型，
# 从而极大地提升性能。这是一种单例模式的应用。
_yolo_model_for_process_img = None
_model_target_class_id_for_process_img = None
_current_process_img_weights_path = None
_current_process_img_target_class = None

# ---------------------------------------------------------------------------- #
#          模块配置接口 (提升模块化)          #
# ---------------------------------------------------------------------------- #
def configure_processor(weights: str = None, conf: float = None, iou: float = None, target_class: str = None):
    """
    动态配置 process_img 函数所使用的全局参数。

    这是推荐的配置方式，它为本模块提供了一个清晰、受控的配置接口，
    避免了其他模块直接修改本模块内部全局变量的紧密耦合问题。

    Args:
        weights: 模型权重文件的路径。
        conf: 目标检测的置信度阈值。
        iou: 用于NMS的IOU阈值。
        target_class: 目标类别名称。
    """
    global DEFAULT_MODEL_WEIGHTS_PATH, DEFAULT_PROCESS_IMG_CONF_THRESHOLD, DEFAULT_PROCESS_IMG_IOU_THRESHOLD, DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME
    if weights is not None:
        DEFAULT_MODEL_WEIGHTS_PATH = weights
    if conf is not None:
        DEFAULT_PROCESS_IMG_CONF_THRESHOLD = conf
    if iou is not None:
        DEFAULT_PROCESS_IMG_IOU_THRESHOLD = iou
    if target_class is not None:
        DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME = target_class


def _initialize_model_for_process_img(
    weights_path: str = DEFAULT_MODEL_WEIGHTS_PATH,
    target_class_name: str = DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME
) -> bool:
    """
    初始化或更新用于 `process_img` 函数的全局 YOLO 模型实例。

    如果请求的权重路径和目标类别与当前已加载的模型相同，则不执行任何操作。
    否则，加载新模型或更新目标类别。

    Args:
        weights_path: 模型权重文件的路径。
        target_class_name: 目标检测的类别名称。空字符串或 None 表示检测所有类别。

    Returns:
        True 如果模型成功加载或已是最新状态，False 如果模型加载失败。
    """
    global _yolo_model_for_process_img, _model_target_class_id_for_process_img
    global _current_process_img_weights_path, _current_process_img_target_class

    abs_weights_path = os.path.abspath(weights_path) if weights_path else None

    # 检查模型是否已加载且配置与请求一致
    if (_yolo_model_for_process_img is not None and
            _current_process_img_weights_path == abs_weights_path and
            _current_process_img_target_class == target_class_name):
        return True

    # 解析权重文件路径 (尝试在CWD, 脚本目录, 项目根目录查找)
    resolved_weights_path = weights_path
    if not os.path.isabs(resolved_weights_path) and not os.path.isfile(resolved_weights_path):
        # ... (路径查找逻辑，为简洁省略，与之前版本相同) ...
        cwd_weights = os.path.join(os.getcwd(), resolved_weights_path)
        if os.path.isfile(cwd_weights): resolved_weights_path = cwd_weights
        else:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                script_dir_weights = os.path.join(script_dir, resolved_weights_path)
                if os.path.isfile(script_dir_weights): resolved_weights_path = script_dir_weights
                else:
                    project_root_weights = os.path.join(os.path.dirname(script_dir), resolved_weights_path)
                    if os.path.isfile(project_root_weights): resolved_weights_path = project_root_weights
            except NameError: # __file__ 可能未定义 (例如在某些交互式环境中)
                pass 
    resolved_weights_path = os.path.abspath(resolved_weights_path) # 确保是绝对路径

    if not os.path.isfile(resolved_weights_path):
        print(f"错误 (_initialize_model_for_process_img): 权重文件 '{resolved_weights_path}' 无效或未找到。")
        _yolo_model_for_process_img = None
        return False

    try:
        print(f"为 process_img 加载/切换模型: {resolved_weights_path}...")
        _yolo_model_for_process_img = YOLO(resolved_weights_path)
        print("模型加载成功 (for process_img).")
        _current_process_img_weights_path = resolved_weights_path
        _current_process_img_target_class = target_class_name

        # 解析目标类别ID
        _model_target_class_id_for_process_img = None
        if target_class_name and target_class_name.strip():
            found_target = False
            for class_id, name in _yolo_model_for_process_img.names.items():
                if name.lower() == target_class_name.strip().lower():
                    _model_target_class_id_for_process_img = class_id
                    found_target = True
                    print(f"process_img 目标类别: '{target_class_name}' (ID: {_model_target_class_id_for_process_img})")
                    break
            if not found_target:
                print(f"警告 (_initialize_model_for_process_img): 在模型中未找到类别 '{target_class_name}'. 将检测所有类别。")
        else:
            print("process_img 未指定目标类别，将检测所有类别.")
        return True
    except Exception as e:
        print(f"模型加载或类别查找失败 (_initialize_model_for_process_img): {e}")
        _yolo_model_for_process_img = None
        return False


def process_img(img_path: str) -> list:
    """
    使用YOLOv8模型处理单张图片并返回检测结果。
    
    此函数符合竞赛/测试模板的要求，仅接受图片路径作为输入。
    模型权重、置信度等参数将使用脚本内定义的全局默认值。

    参数:
       img_path: 要识别的图片的路径。
    
    返回:
       一个检测结果列表。每个元素是一个字典，格式如下:
        {
            'x': '目标框左上角x坐标 (int)',
            'y': '目标框左上角y坐标 (int)',
            'w': '目标框宽度 (int)',
            'h': '目标框高度 (int)'
        }
    """
    # 使用脚本顶部的全局默认配置
    current_weights = DEFAULT_MODEL_WEIGHTS_PATH
    current_conf = DEFAULT_PROCESS_IMG_CONF_THRESHOLD
    current_iou = DEFAULT_PROCESS_IMG_IOU_THRESHOLD
    current_target_class = DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME

    if not _initialize_model_for_process_img(weights_path=current_weights, target_class_name=current_target_class):
        print(f"错误 (process_img): 模型未能为路径 '{current_weights}' 和目标 '{current_target_class}' 初始化。")
        return []

    if not os.path.isfile(img_path):
        print(f"错误 (process_img): 图片文件 '{img_path}' 未找到。")
        return []

    detections = []
    try:
        # verbose=False 减少 predict 函数的控制台输出
        results = _yolo_model_for_process_img.predict(source=img_path, conf=current_conf, iou=current_iou, verbose=False)

        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if _model_target_class_id_for_process_img is not None and class_id != _model_target_class_id_for_process_img:
                    continue
                
                xyxy_val = box.xyxy[0].cpu().numpy().astype(int)
                x_min, y_min, x_max, y_max = xyxy_val
                width, height = x_max - x_min, y_max - y_min

                detections.append({
                    'x': int(x_min),
                    'y': int(y_min),
                    'w': int(width),
                    'h': int(height)
                })
    except Exception as e:
        print(f"处理图片 '{img_path}' 时发生错误 (process_img): {e}")
    return detections


def draw_detections(frame, detections: list) -> any:
    """
    在图片上绘制检测框和标签。
    此函数从 run_cli.py 复制而来，用于在 process.py 中进行结果可视化。
    """
    for det in detections:
        # 传入的 detections 格式为: {'label': ..., 'confidence': ..., 'bbox': [x, y, w, h]}
        label = det.get("label", "unknown")
        confidence = det.get("confidence", -1.0) # 使用-1表示置信度不可用
        x_min, y_min, width, height = det.get("bbox", [0,0,0,0])
        x_max, y_max = x_min + width, y_min + height
        
        # 绘制矩形框
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 准备标签文本
        if confidence >= 0 and confidence != 1.0: # 仅当置信度有效且不是伪造的1.0时显示
            label_text = f"{label} {confidence:.2f}"
        else:
            label_text = label # 如果置信度不可用，则只显示标签

        # 绘制标签背景和文字
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x_min, y_min - h - 10), (x_min + w, y_min - 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame

# ---------------------------------------------------------------------------- #
#          主程序入口 (仅当直接运行时执行)          #
# ---------------------------------------------------------------------------- #
if __name__=='__main__':
    # 这部分代码块提供了一个方便的、自包含的测试功能。
    # 当你通过 `python src/process.py` 直接运行此文件时，
    # 它会自动处理 `imgs` 目录下的所有图片，并将结果（包括标注图片和数据文件）
    # 保存到 `results/process_py_output` 目录中。
    # 这个功能不会在 `process.py` 被其他脚本作为模块导入时触发。
    
    # --- 配置 ---
    imgs_folder = './imgs/'
    output_base_dir = "results/process_py_output" # 定义一个独立的输出目录
    
    print(f"--- 直接运行 process.py (带自动保存功能) ---")
    print(f"输入目录: {imgs_folder}")
    print(f"输出将保存到: {output_base_dir}")

    # 检查测试文件夹是否存在
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder)
        print(f"测试文件夹 '{imgs_folder}' 不存在，已创建。")

    # 获取所有图片路径
    img_paths = [os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_paths:
        print(f"错误: 测试文件夹 '{imgs_folder}' 为空。请将您的测试图片放入该文件夹中再重新运行。")
    else:
        # --- 计时变量 ---
        def now(): return int(time.time() * 1000)
        count_time = 0
        max_time = 0
        min_time = float('inf')
        
        # --- 处理循环 ---
        for img_path in img_paths:
            print(f"\n--- 正在处理: {os.path.basename(img_path)} ---")
            
            last_time = now()
            
            # 1. 调用核心检测函数
            detections = process_img(img_path)
            
            run_time = now() - last_time
            
            print(f"检测到 {len(detections)} 个目标。")
            print(f"运行时间: {run_time} ms")

            # 更新计时
            count_time += run_time
            if run_time > max_time: max_time = run_time
            if run_time < min_time: min_time = run_time

            # 2. 准备输出目录
            img_stem = Path(img_path).stem
            current_output_dir = os.path.join(output_base_dir, img_stem)
            os.makedirs(current_output_dir, exist_ok=True)
            
            # 3. 保存JSON结果 (格式与 run_cli.py 保持一致)
            json_path = os.path.join(current_output_dir, f"{img_stem}_results.txt")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detections, f, ensure_ascii=False, indent=4)
            print(f"检测结果已保存至: {json_path}")
            
            # 4. 保存标注图片
            frame = cv2.imread(img_path)
            
            # 转换检测结果格式以适配绘图函数 (模仿 run_cli.py 的做法)
            temp_detections_for_drawing = []
            for det in detections:
                temp_detections_for_drawing.append({
                    "label": DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME or "detection",
                    "confidence": -1, # process_img不返回置信度，设为-1
                    "bbox": [det['x'], det['y'], det['w'], det['h']]
                })
                
            annotated_frame = draw_detections(frame, temp_detections_for_drawing)
            
            save_path = os.path.join(current_output_dir, f"annotated_{os.path.basename(img_path)}")
            cv2.imwrite(save_path, annotated_frame)
            print(f"标注图片已保存至: {save_path}")
            
        # --- 打印统计信息 ---
        if len(img_paths) > 0:
            print('\n--- 统计信息 ---')
            print(f'平均时间: {int(count_time / len(img_paths))} ms')
            print(f'最长时间: {max_time} ms')
            print(f'最短时间: {min_time} ms')