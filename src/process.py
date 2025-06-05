# -*- coding: utf-8 -*-
"""
YOLOv8 目标检测脚本。

该脚本提供了两种主要功能：
1. 通过命令行接口 (CLI) 使用 `detect_tennis_ball` 函数进行目标检测。
   支持对单个图片、视频文件或图片目录进行检测，并提供多种输出选项
   （显示、保存标注图片/视频、保存JSON结果）。
2. 通过 `process_img` 函数处理单张图片并返回结构化的检测结果。
   此函数设计为可以被其他 Python 模块导入和调用，或者在脚本无命令行参数
   直接运行时进行测试。

主要依赖: ultralytics, opencv-python, numpy.
"""

import os
import time
import cv2
from ultralytics import YOLO
import argparse
import json
import sys
import numpy as np
from pathlib import Path

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


def process_img(img_path: str,
                weights_path: str = None,
                conf_threshold: float = None,
                iou_threshold: float = None,
                target_class_name: str = None) -> list:
    """
    使用YOLOv8模型处理单张图片并返回检测结果。

    此函数可以被其他模块导入和调用。它使用全局缓存的模型实例以提高效率。
    可以通过参数临时覆盖全局默认的模型权重、阈值和目标类别。

    Args:
        img_path: 要识别的图片的路径。
        weights_path: (可选) 模型权重路径。若为None，则使用全局默认值。
        conf_threshold: (可选) 置信度阈值。若为None，则使用全局默认值。
        iou_threshold: (可选) IOU阈值。若为None，则使用全局默认值。
        target_class_name: (可选) 目标类别名称。若为None，则使用全局默认值。

    Returns:
        一个检测结果列表。每个元素是一个字典，格式如下:
        {
            "label": "识别出的物体类别名 (str)",
            "confidence": "置信度 (float)",
            "bbox": "[x_min, y_min, width, height] (list of int)"
        }
        如果未检测到物体或发生错误，返回空列表。
    """
    # 确定本次调用使用的配置
    current_weights = weights_path if weights_path is not None else DEFAULT_MODEL_WEIGHTS_PATH
    current_conf = conf_threshold if conf_threshold is not None else DEFAULT_PROCESS_IMG_CONF_THRESHOLD
    current_iou = iou_threshold if iou_threshold is not None else DEFAULT_PROCESS_IMG_IOU_THRESHOLD
    current_target_class = target_class_name if target_class_name is not None else DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME

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

        # predict 对于单个图片返回一个包含单个Results对象的列表
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                # 如果设定了目标类别ID，则过滤掉不匹配的检测
                if _model_target_class_id_for_process_img is not None and class_id != _model_target_class_id_for_process_img:
                    continue
                
                xyxy_val = box.xyxy[0].cpu().numpy().astype(int)
                x_min, y_min, x_max, y_max = xyxy_val
                width, height = x_max - x_min, y_max - y_min

                detections.append({
                    "label": _yolo_model_for_process_img.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [x_min, y_min, width, height] # x_min, y_min, w, h
                })
    except Exception as e:
        print(f"处理图片 '{img_path}' 时发生错误 (process_img): {e}")
    return detections


# ---------------------------------------------------------------------------- #
#                            命令行接口相关函数                            #
# ---------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，用于 `detect_tennis_ball` 函数。
    """
    parser = argparse.ArgumentParser(
        description="YOLOv8 目标检测脚本 (命令行工具模式)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # 显示默认值
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_MODEL_WEIGHTS_PATH,
        help="模型权重文件的路径。"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="输入源：单个图片路径、视频文件路径、摄像头ID (如 '0') 或包含图片的目录路径。"
    )
    parser.add_argument(
        "--output_dir", type=str, default="runs/detect/exp_cli",
        help="检测结果的主输出目录。如果输入是图片目录，将在此目录下为每张图片创建子目录。"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=DEFAULT_PROCESS_IMG_CONF_THRESHOLD,
        help="目标检测的置信度阈值。"
    )
    parser.add_argument(
        "--iou_thres", type=float, default=DEFAULT_PROCESS_IMG_IOU_THRESHOLD,
        help="用于NMS (非极大值抑制) 的IOU阈值。"
    )
    parser.add_argument(
        "--show_vid", action="store_true",
        help="是否实时显示处理后的视频流或图片。"
    )
    parser.add_argument(
        "--save_vid", action="store_true",
        help="是否保存处理后的视频。若输入源是图片或目录，则保存标注后的图片。"
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="是否为每个处理的输入图片生成包含检测结果的JSON格式 .txt 文件。"
             "文件将保存在对应图片的子输出目录中。"
    )
    parser.add_argument(
        "--output_json_path", type=str, default="", # 此参数基本已弃用
        help="[已弃用] 此参数在当前 'save_json' 模式下被忽略，因结果文件在输出目录的子目录中生成。"
    )
    parser.add_argument(
        "--target_class", type=str, default=DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME,
        help="要专门检测的目标类别名称 (例如 'tennis ball')。若为空或未指定，则检测所有类别。"
    )
    return parser.parse_args()


def detect_tennis_ball(weights_path_cli: str, source: str, output_dir_base: str,
                       conf_thres: float, iou_thres: float,
                       show_vid: bool, save_vid: bool,
                       save_json: bool, output_json_path: str, # output_json_path is largely ignored
                       target_class_name: str) -> int:
    """
    使用YOLOv8模型通过命令行接口进行目标检测。

    可以处理单个图片文件、视频文件或包含图片的目录。
    结果（标注图片、JSON .txt 文件）会根据输入源的类型和指定的输出目录进行组织。

    Args:
        weights_path_cli: 模型权重路径。
        source: 输入源路径 (文件、目录、摄像头ID、URL)。
        output_dir_base: 检测结果的主输出目录。
        conf_thres: 置信度阈值。
        iou_thres: IOU阈值。
        show_vid: 是否实时显示结果。
        save_vid: 是否保存标注后的图片/视频。
        save_json: 是否为每个图片保存JSON格式的 .txt 结果文件。
        output_json_path: (基本已弃用) 旧的聚合JSON输出路径。
        target_class_name: 目标类别名称。

    Returns:
        执行状态码：0 表示成功，1 表示发生错误。
    """
    print(f"\n--- detect_tennis_ball (命令行模式) ---")
    print(f"  权重: {weights_path_cli}, 源: {source}")
    print(f"  配置: conf={conf_thres}, iou={iou_thres}, 目标类='{target_class_name}'")
    print(f"  输出: 主目录='{output_dir_base}', 显示={show_vid}, 保存标注={save_vid}, 保存JSON TXT={save_json}")

    # 解析权重路径
    cli_resolved_weights_path = weights_path_cli
    if not os.path.isabs(cli_resolved_weights_path) and not os.path.isfile(cli_resolved_weights_path):
        # ... (权重路径解析逻辑，为简洁省略，与之前版本相同) ...
        cwd_weights = os.path.join(os.getcwd(), cli_resolved_weights_path)
        if os.path.isfile(cwd_weights): cli_resolved_weights_path = cwd_weights
    if not os.path.isfile(cli_resolved_weights_path):
        print(f"错误 (CLI): 最终权重文件 '{cli_resolved_weights_path}' 未找到。")
        return 1

    # 加载模型
    print(f"正在加载模型 (CLI): {cli_resolved_weights_path}...")
    try:
        model_cli = YOLO(cli_resolved_weights_path)
        print("模型加载成功 (CLI).")
        # ... (目标类ID查找逻辑，为简洁省略，与之前版本相同) ...
        cli_target_class_id = None
        if target_class_name and target_class_name.strip():
            found_target = False
            for class_id, name_val in model_cli.names.items():
                if name_val.lower() == target_class_name.strip().lower():
                    cli_target_class_id = class_id; found_target = True; break
            if found_target: print(f"CLI 目标类别: '{target_class_name}' (ID: {cli_target_class_id})")
            else: print(f"警告 (CLI): 未找到类别 '{target_class_name}'.")
        else: print("CLI 未指定目标类别，将检测所有类别.")
    except Exception as e:
        print(f"模型加载或类别查找失败 (CLI): {e}")
        return 1

    # 进行预测
    print(f"正在对源进行预测 (CLI): {source}...")
    try:
        results_generator = model_cli.predict(source, stream=True, conf=conf_thres, iou=iou_thres, imgsz=640, verbose=False)
    except Exception as e:
        print(f"预测过程中发生错误 (CLI): {e}")
        if not os.path.exists(source): print(f"错误: 输入源 '{source}' 不存在。")
        return 1

    # 判断输入源类型
    is_input_video = False; is_input_directory = False; is_input_single_image = False
    # ... (输入源类型判断逻辑，为简洁省略，与之前版本相同) ...
    if os.path.isdir(source): is_input_directory = True
    elif os.path.isfile(source):
        source_ext = Path(source).suffix.lower()
        if source_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']: is_input_single_image = True
        elif source_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg']: is_input_video = True


    # 创建主输出目录
    os.makedirs(output_dir_base, exist_ok=True)

    print("开始处理检测结果 (CLI)...")
    video_writer = None
    first_frame_for_video = True

    # 遍历检测结果
    for result in results_generator:
        original_item_path_str = result.path
        item_stem = "unknown_item"; item_fullname = "Stream" # 默认值
        # ... (item_stem 和 item_fullname 的确定逻辑，为简洁省略，与之前版本相同) ...
        if original_item_path_str and isinstance(original_item_path_str, str):
            item_stem = Path(original_item_path_str).stem
            item_fullname = Path(original_item_path_str).name
        elif is_input_video: 
            frame_idx = result.count if hasattr(result, 'count') else str(time.time_ns())[-6:]
            item_stem = f"{Path(source).stem}_frame_{frame_idx}"
            item_fullname = item_stem

        # 为当前图片创建子输出目录 (视频帧不创建单独子目录)
        current_item_output_dir = output_dir_base
        if is_input_directory or is_input_single_image:
            current_item_output_dir = os.path.join(output_dir_base, item_stem)
            os.makedirs(current_item_output_dir, exist_ok=True)

        annotated_frame = result.orig_img.copy()
        current_item_json_data = [] # 用于存储当前项的检测数据
        log_boxes = []

        # 处理每个检测框
        for box in result.boxes:
            class_id = int(box.cls[0])
            if cli_target_class_id is not None and class_id != cli_target_class_id:
                continue
            
            conf_val = float(box.conf[0])
            xyxy_val = box.xyxy[0].cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = xyxy_val
            width, height = x_max - x_min, y_max - y_min

            # 绘制到帧上
            label_txt = f"{model_cli.names[class_id]} {conf_val:.2f}"
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label_txt, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 收集JSON数据
            current_item_json_data.append({
                "label": model_cli.names[class_id],
                "confidence": round(conf_val, 4),
                "bbox": [x_min, y_min, width, height]
            })
            log_boxes.append((model_cli.names[class_id], conf_val, [x_min, y_min, x_max, y_max]))

        # 显示结果
        if show_vid:
            cv2.imshow(f"YOLOv8 Detection - {item_fullname}", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                show_vid = False  # 停止后续显示
                break             # 退出结果处理循环

        # 保存标注后的图片/视频
        if save_vid:
            if is_input_video:
                # ... (视频保存逻辑，为简洁省略，与之前版本相同，保存在 output_dir_base) ...
                if first_frame_for_video:
                    h_vid, w_vid = annotated_frame.shape[:2]; fps_vid = 30
                    cap_temp = cv2.VideoCapture(source); fps_tmp = cap_temp.get(cv2.CAP_PROP_FPS); cap_temp.release()
                    if fps_tmp > 0: fps_vid = fps_tmp
                    vid_out_p = os.path.join(output_dir_base, f"processed_{Path(source).name}") #视频在主输出目录
                    video_writer = cv2.VideoWriter(vid_out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (w_vid,h_vid))
                    print(f"保存视频到: {vid_out_p}"); first_frame_for_video = False
                if video_writer: video_writer.write(annotated_frame)
            elif is_input_directory or is_input_single_image:
                # 标注图片保存在其子目录 current_item_output_dir
                img_save_filename = f"result_{item_stem}.jpg"
                img_save_path = os.path.join(current_item_output_dir, img_save_filename)
                cv2.imwrite(img_save_path, annotated_frame)

        # 保存当前图片的JSON .txt 文件
        if save_json:
            if is_input_directory or is_input_single_image:
                # JSON .txt 文件保存在其子目录 current_item_output_dir
                individual_txt_filename = f"{item_stem}.txt"
                individual_txt_path = os.path.join(current_item_output_dir, individual_txt_filename)
                try:
                    with open(individual_txt_path, 'w', encoding='utf-8') as f_txt:
                        json.dump(current_item_json_data, f_txt, ensure_ascii=False, indent=4)
                except Exception as e_txt:
                    print(f"  保存单个TXT文件 '{individual_txt_path}' 失败: {e_txt}")
            # 注意：当前未给视频的每一帧保存单独的JSON TXT文件

        # 打印日志
        print(f"--- {item_fullname} (CLI) ---")
        # ... (日志打印逻辑，为简洁省略，与之前版本相同) ...
        if log_boxes: [print(f"  检测到: {n}, 置信度: {c:.2f}, BBox (xyxy): {b}") for n,c,b in log_boxes]
        else: print("  未检测到符合条件的目标。")

    # 循环结束后清理
    if video_writer:
        video_writer.release()
        print("视频已保存 (CLI).")
    
    # 尝试关闭所有OpenCV窗口
    # cv2.waitKey(1) 确保窗口事件被处理，然后 destroyAllWindows 才能生效
    if cv2.getWindowProperty(f"YOLOv8 Detection - {item_fullname}",0) >=0 or show_vid: # 检查窗口是否可能还存在
        cv2.waitKey(1) # 给窗口一点时间响应
        cv2.destroyAllWindows()
        cv2.waitKey(1) # 再一次确保关闭

    print("检测处理完毕 (CLI).")
    return 0 # 表示成功


# ---------------------------------------------------------------------------- #
#                                 主执行块                                   #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 判断是否通过命令行参数调用
    is_cli_call = any(arg.startswith('--') for arg in sys.argv[1:])
    
    if is_cli_call:
        print("检测到命令行参数，运行 CLI 工具 (detect_tennis_ball)...")
        args = parse_args()

        if args.save_json and args.output_json_path:
            print(f"信息: --output_json_path ('{args.output_json_path}') 被指定，但在当前 'save_json' "
                  f"模式下，结果会为每个输入图片在 --output_dir 的子目录中生成单独的 .txt 文件。"
                  f"此参数将被忽略。")

        # 执行核心检测函数
        exit_status = detect_tennis_ball(
            weights_path_cli=args.weights,
            source=args.source,
            output_dir_base=args.output_dir, # 传递给函数作为主输出目录
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            show_vid=args.show_vid,
            save_vid=args.save_vid,
            save_json=args.save_json,
            output_json_path=args.output_json_path, # 传递但在此模式下主要被忽略
            target_class_name=args.target_class
        )
        sys.exit(exit_status) # 将函数的返回状态作为脚本的退出状态
    else:
        # 如果没有命令行参数，则执行 process_img 的测试流程
        print("--- 未检测到命令行参数，开始测试 process_img 函数 ---")
        # ... (process_img 的测试逻辑，为简洁省略，与之前版本相同) ...
        test_imgs_folder = './imgs/' 
        if not os.path.exists(test_imgs_folder): os.makedirs(test_imgs_folder, exist_ok=True)
        dummy_img_path = os.path.join(test_imgs_folder, "dummy_test_image.png")
        if not os.path.exists(dummy_img_path):
            try: cv2.imwrite(dummy_img_path, np.zeros((480, 640, 3), np.uint8))
            except Exception as e: print(f"创建虚拟图出错: {e}")
        img_paths_to_test = [os.path.join(test_imgs_folder, f) for f in os.listdir(test_imgs_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_paths_to_test and os.path.exists(dummy_img_path): img_paths_to_test.append(dummy_img_path)
        if not img_paths_to_test: print(f"警告: 测试图片文件夹 '{test_imgs_folder}' 为空。")
        else:
            for img_p_test in img_paths_to_test:
                print(f"\n处理图片 (process_img test): {img_p_test}")
                res = process_img(img_p_test) 
                print(f"结果 ({len(res)} 个检测):")
                if res: [print(f"  - {item}") for item in res]
                else: print("  未检测到物体。")
        print("--- process_img 函数测试结束 ---")
        sys.exit(0)