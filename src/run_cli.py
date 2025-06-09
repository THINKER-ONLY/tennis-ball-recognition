# -*- coding: utf-8 -*-
"""
命令行接口 (CLI) 的桥梁脚本。

该脚本旨在恢复 `process.py` 在被修改为简单API模板之前所具备的丰富命令行功能。
它接收命令行参数，并调用 `src.process.py` 中的 `process_img` 函数来执行核心检测任务。
它处理不同类型的输入源（图片、目录、视频），并管理结果的显示与保存。

主要功能:
1. 解析与旧版 `process.py` 兼容的命令行参数。
2. 处理单个图片、图片目录或视频文件。
3. 调用 `process_img` 函数进行目标检测。
4. 在屏幕上显示处理结果或将标注后的图片/视频保存到磁盘。
5. 保存JSON格式的检测结果。
"""
import os
import sys
import cv2
import argparse
import json
from pathlib import Path
import time

# 将项目根目录添加到Python路径中，以便可以正确导入src模块
# 这使得脚本可以从任何位置运行
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from src.process import process_img, DEFAULT_MODEL_WEIGHTS_PATH, DEFAULT_PROCESS_IMG_CONF_THRESHOLD, DEFAULT_PROCESS_IMG_IOU_THRESHOLD, DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME
except ImportError as e:
    print(f"错误: 无法导入 'src.process' 模块。请确保脚本位于项目的 'src' 文件夹下。({e})")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 目标检测的命令行桥梁。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights", type=str, default=DEFAULT_MODEL_WEIGHTS_PATH,
        help="模型权重文件的路径。"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="输入源：单个图片路径、视频文件路径或包含图片的目录路径。"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/detection_output",
        help="检测结果的主输出目录。"
    )
    parser.add_argument(
        "--conf_thres", type=float, default=DEFAULT_PROCESS_IMG_CONF_THRESHOLD,
        help="目标检测的置信度阈值。"
    )
    parser.add_argument(
        "--iou_thres", type=float, default=DEFAULT_PROCESS_IMG_IOU_THRESHOLD,
        help="用于NMS的IOU阈值。"
    )
    parser.add_argument(
        "--show_vid", action="store_true",
        help="是否实时显示处理后的视频流或图片。"
    )
    parser.add_argument(
        "--save_vid", action="store_true",
        help="是否保存处理后的视频或标注后的图片。"
    )
    parser.add_argument(
        "--save_json", action="store_true",
        help="是否为每个处理的图片生成包含检测结果的JSON .txt文件。"
    )
    parser.add_argument(
        "--target_class", type=str, default=DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME,
        help="要专门检测的目标类别名称。若为空，则检测所有类别。"
    )
    # 这个参数在我们的新结构中不再需要，但为了与run.sh兼容，暂时保留
    parser.add_argument(
        "--output_json_path", type=str, default="",
        help="[已弃用] 此参数被忽略。"
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    """根据解析的参数执行检测流程。"""
    print("--- 命令行桥梁脚本 (run_cli.py) 开始执行 ---")

    # ---- 1. 确定输入源类型 ----
    source_path = args.source
    is_video = False
    is_directory = False
    is_single_image = False

    if not os.path.exists(source_path):
        print(f"错误: 输入源 '{source_path}' 不存在。")
        return

    if os.path.isdir(source_path):
        is_directory = True
        image_paths = sorted([os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))])
        print(f"检测到输入为目录，共找到 {len(image_paths)} 张图片。")
    elif os.path.isfile(source_path):
        ext = Path(source_path).suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
            print("检测到输入为视频文件。")
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            is_single_image = True
            image_paths = [source_path]
            print("检测到输入为单个图片。")
        else:
            print(f"错误: 不支持的文件类型 '{ext}'。")
            return
    
    # ---- 2. 准备输出 ----
    output_dir = args.output_dir
    if args.save_vid or args.save_json:
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存到目录: {output_dir}")

    video_writer = None

    # ---- 3. 处理输入 ----
    
    # A. 如果是视频
    if is_video:
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 '{source_path}'。")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 由于 process_img 需要文件路径，我们将帧临时保存到磁盘
            temp_frame_path = os.path.join(output_dir, f"temp_frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)

            # 调用核心函数进行检测
            detections = process_img(temp_frame_path)
            
            # 在帧上绘制检测结果
            annotated_frame = draw_detections(frame, detections)

            # 显示或保存
            if args.show_vid:
                cv2.imshow("Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if args.save_vid:
                if video_writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = os.path.join(output_dir, f"processed_{Path(source_path).name}")
                    video_writer = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
                video_writer.write(annotated_frame)
            
            # 清理临时文件
            os.remove(temp_frame_path)
            frame_count += 1
        
        cap.release()
        if video_writer:
            video_writer.release()

    # B. 如果是图片或目录
    elif is_directory or is_single_image:
        for img_path in image_paths:
            print(f"\n--- 正在处理: {Path(img_path).name} ---")
            
            # 为当前图片创建独立的子输出目录
            img_stem = Path(img_path).stem
            current_output_dir = os.path.join(output_dir, img_stem)
            if args.save_vid or args.save_json:
                os.makedirs(current_output_dir, exist_ok=True)
                print(f"当前图片的输出将保存到: {current_output_dir}")

            # 调用核心函数
            detections = process_img(img_path)
            print(f"检测到 {len(detections)} 个目标。")

            # 保存JSON结果
            if args.save_json:
                json_path = os.path.join(current_output_dir, f"{img_stem}.txt")
                with open(json_path, 'w', encoding='utf-8') as f:
                    # 注意：虽然标志是 save_json，但实际保存的是 txt 文件
                    # 为了与竞赛要求保持一致，这里我们直接写入 process_img 返回的列表
                    json.dump(detections, f, ensure_ascii=False, indent=4)
                print(f"结果已保存至: {json_path}")
            
            # 显示或保存标注图片
            if args.show_vid or args.save_vid:
                frame = cv2.imread(img_path)
                
                # draw_detections 现在需要使用不同的 bbox 格式
                # 我们需要修改它或在这里适配
                # 让我们在这里创建一个临时适配的版本
                temp_detections_for_drawing = []
                for det in detections:
                    temp_detections_for_drawing.append({
                        "label": "tennis ball", # 标签可以硬编码或从配置中获取
                        "confidence": 1.0, # 置信度在当前格式下不可用
                        "bbox": [det['x'], det['y'], det['w'], det['h']]
                    })
                annotated_frame = draw_detections(frame, temp_detections_for_drawing)

                if args.show_vid:
                    cv2.imshow(f"Detection - {Path(img_path).name}", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if args.save_vid:
                    save_path = os.path.join(current_output_dir, f"processed_{Path(img_path).name}")
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"标注图片已保存至: {save_path}")

    if not args.show_vid:
        cv2.destroyAllWindows()
    print("--- 命令行桥梁脚本 (run_cli.py) 执行完毕 ---")


def draw_detections(frame, detections: list) -> any:
    """在图片上绘制检测框和标签。"""
    for det in detections:
        label = det.get("label", "unknown")
        confidence = det.get("confidence", 0)
        x_min, y_min, width, height = det.get("bbox", [0,0,0,0])
        x_max, y_max = x_min + width, y_min + height
        
        # 绘制矩形框
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 准备标签文本
        label_text = f"{label} {confidence:.2f}"
        
        # 绘制标签背景和文字
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x_min, y_min - h - 10), (x_min + w, y_min - 5), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame


if __name__ == "__main__":
    cli_args = parse_args()
    
    # 将模型权重和超参数设置到process模块中
    # 这是必要的，因为process_img()没有这些参数
    # 我们需要一个方法来把CLI的参数传递进去
    # (注意：这是对原process.py的一个小入侵，但对于功能是必要的)
    try:
        from src import process
        process.DEFAULT_MODEL_WEIGHTS_PATH = cli_args.weights
        process.DEFAULT_PROCESS_IMG_CONF_THRESHOLD = cli_args.conf_thres
        process.DEFAULT_PROCESS_IMG_IOU_THRESHOLD = cli_args.iou_thres
        process.DEFAULT_PROCESS_IMG_TARGET_CLASS_NAME = cli_args.target_class
        print("已通过CLI参数动态更新process.py中的默认配置。")
    except Exception as e:
        print(f"警告: 动态更新 process.py 配置失败: {e}")

    run(cli_args) 