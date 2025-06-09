# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测模块。

该脚本的核心是 `process_img` 函数，它接收一张图片路径，
并返回图中检测到的物体的结构化数据。

主要依赖: ultralytics, opencv-python, numpy.
"""

import os
import time
import cv2
from ultralytics import YOLO
import numpy as np

# ---------------------------------------------------------------------------- #
#          默认配置 (主要用于 process_img 函数的直接调用)          #
# ---------------------------------------------------------------------------- #
DEFAULT_MODEL_WEIGHTS_PATH = "yolov8n.pt"  # process_img 使用的默认模型权重
DEFAULT_PROCESS_IMG_CONF_THRESHOLD = 0.25  # process_img 使用的默认置信度阈值
DEFAULT_PROCESS_IMG_IOU_THRESHOLD = 0.45   # process_img 使用的默认IOU阈值
# process_img 使用的默认目标类别。设为空字符串 "" 或 None 则检测所有类别。
DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME = "tennis ball"

# 全局变量，用于缓存 process_img 函数加载的模型及其配置，避免重复加载。
_yolo_model_for_process_img = None
_model_target_class_id_for_process_img = None
_current_process_img_weights_path = None
_current_process_img_target_class = None


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


#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
if __name__=='__main__':
    imgs_folder = './imgs/'
    # 检查测试文件夹是否存在，不存在则创建
    if not os.path.exists(imgs_folder):
        print(f"测试文件夹 '{imgs_folder}' 不存在，正在创建...")
        os.makedirs(imgs_folder)
        # 创建一张虚拟图片以提示用户
        dummy_img_path = os.path.join(imgs_folder, "dummy_test_image.png")
        try:
            cv2.imwrite(dummy_img_path, np.zeros((480, 640, 3), dtype=np.uint8))
            print(f"已在 '{imgs_folder}' 中创建一张虚拟测试图片。")
            print("请将您的测试图片放入该文件夹中再重新运行。")
        except Exception as e:
            print(f"创建虚拟图片失败: {e}")

    # 获取所有图片路径
    img_paths = [os.path.join(imgs_folder, f) for f in os.listdir(imgs_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_paths:
        print(f"测试文件夹 '{imgs_folder}' 为空，无法进行测试。")
    else:
        def now():
            return int(time.time()*1000)
        
        count_time = 0
        max_time = 0
        min_time = float('inf') # 使用极大值初始化，确保第一次比较时被覆盖
        
        for img_path in img_paths:
            print(img_path,':')
            last_time = now()
            result = process_img(img_path)
            run_time = now() - last_time
            print('result:\n',result)
            print('run time: ', run_time, 'ms')
            print()
            count_time += run_time
            if run_time > max_time:
                max_time = run_time
            if run_time < min_time:
                min_time = run_time
        
        if len(img_paths) > 0:
            print('\n')
            print('avg time: ',int(count_time/len(img_paths)),'ms')
            print('max time: ',max_time,'ms')
            print('min time: ',min_time,'ms')