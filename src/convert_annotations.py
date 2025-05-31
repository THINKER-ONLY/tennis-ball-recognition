import os
import json
import cv2
import argparse
from tqdm import tqdm # 用于显示进度条

def convert_bbox_to_yolo(img_width, img_height, bbox_x, bbox_y, bbox_w, bbox_h, class_id=0):
    """
    将单个边界框从 (左上角x, 左上角y, 宽, 高) 绝对像素值转换为YOLO格式。

    参数:
        img_width (int): 图像宽度。
        img_height (int): 图像高度。
        bbox_x (int): 边界框左上角 x 坐标 (绝对像素)。
        bbox_y (int): 边界框左上角 y 坐标 (绝对像素)。
        bbox_w (int): 边界框宽度 (绝对像素)。
        bbox_h (int): 边界框高度 (绝对像素)。
        class_id (int): 类别 ID (默认为0)。

    返回:
        str: YOLO格式的标注字符串 "class_id x_center_norm y_center_norm width_norm height_norm"。
             如果边界框无效或超出图像范围，则可能返回 None 或进行裁剪处理（当前未实现裁剪）。
    """
    # 检查边界框有效性 (非常重要，避免除以零或负尺寸)
    if bbox_w <= 0 or bbox_h <= 0 or img_width <= 0 or img_height <= 0:
        print(f"警告: 无效的边界框或图像尺寸。bbox_w={bbox_w}, bbox_h={bbox_h}, img_w={img_width}, img_h={img_height}")
        return None

    dw = 1.0 / img_width
    dh = 1.0 / img_height

    x_center = bbox_x + bbox_w / 2.0
    y_center = bbox_y + bbox_h / 2.0

    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    width_norm = bbox_w * dw
    height_norm = bbox_h * dh

    # 确保归一化值在 [0, 1] 范围内 (YOLO期望)
    # 可以根据需要添加更严格的裁剪逻辑，例如确保 x_center_norm - width_norm/2 >= 0 等
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    # 如果归一化后的宽度或高度为0，说明原始框可能非常小或在边缘，也视为无效
    if width_norm == 0 or height_norm == 0:
        # print(f"警告: 归一化后宽度/高度为0。bbox_w={bbox_w}, bbox_h={bbox_h}, img_w={img_width}, img_h={img_height}")
        return None

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def process_annotations(json_file_path, images_dir_path, output_labels_dir_path, default_class_id=0):
    """
    读取JSON标注文件，转换为YOLO格式，并保存到输出目录。
    """
    print(f"正在加载JSON标注文件: {json_file_path}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: JSON文件未找到 at {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: JSON文件格式无效 at {json_file_path}")
        return
    except Exception as e:
        print(f"读取JSON文件时发生未知错误: {e}")
        return

    if not os.path.exists(images_dir_path):
        print(f"错误: 图片目录未找到 at {images_dir_path}")
        return

    os.makedirs(output_labels_dir_path, exist_ok=True)
    print(f"YOLO标注将保存到: {output_labels_dir_path}")

    conversion_errors = 0
    successful_conversions = 0
    images_processed = 0

    print("开始转换标注...")
    # 使用tqdm显示进度条
    for image_filename, bboxes in tqdm(annotations_data.items(), desc="处理图片中"):
        image_path = os.path.join(images_dir_path, image_filename)

        if not os.path.exists(image_path):
            print(f"警告: 图片文件 {image_filename} 在目录 {images_dir_path} 中未找到，跳过。")
            conversion_errors += len(bboxes) if isinstance(bboxes, list) else 1
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: OpenCV无法读取图片 {image_filename}，跳过。")
                conversion_errors += len(bboxes) if isinstance(bboxes, list) else 1
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"读取图片 {image_filename} 尺寸时出错: {e}，跳过。")
            conversion_errors += len(bboxes) if isinstance(bboxes, list) else 1
            continue
        
        images_processed += 1
        yolo_annotations_for_image = []

        if not isinstance(bboxes, list):
            # print(f"信息: 图片 {image_filename} 没有有效的边界框列表 (可能是空列表的错误表示方式)。")
            # 即使是空的，也创建一个空的label文件，表示该图片没有目标物体
            pass # 会在下面创建空文件
        else:
            for bbox_data in bboxes:
                try:
                    x, y, w, h = bbox_data['x'], bbox_data['y'], bbox_data['w'], bbox_data['h']
                    yolo_format_str = convert_bbox_to_yolo(img_width, img_height, x, y, w, h, default_class_id)
                    if yolo_format_str:
                        yolo_annotations_for_image.append(yolo_format_str)
                        successful_conversions +=1
                    else:
                        # print(f"信息: 图片 {image_filename} 中的一个边界框无法转换或无效: {bbox_data}")
                        conversion_errors += 1 # 即使返回None也算一个尝试过的框
                except KeyError as ke:
                    print(f"警告: 图片 {image_filename} 的标注中缺少键: {ke}。标注: {bbox_data}。跳过此框。")
                    conversion_errors += 1
                except Exception as ex_bbox:
                    print(f"警告: 处理图片 {image_filename} 的一个边界框时出错: {ex_bbox}。标注: {bbox_data}。跳过此框。")
                    conversion_errors += 1

        # 为每张图片创建一个.txt文件，即使它没有标注 (YOLO期望如此)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_file_path = os.path.join(output_labels_dir_path, label_filename)

        try:
            with open(label_file_path, 'w', encoding='utf-8') as f_label:
                for line in yolo_annotations_for_image:
                    f_label.write(line + "\n")
        except Exception as e_write:
            print(f"错误: 无法写入标注文件 {label_file_path}: {e_write}")
            # 这个错误比较严重，可能需要停止或特殊处理
            conversion_errors += len(yolo_annotations_for_image) # 如果写入失败，这些转换也算失败
            successful_conversions -= len(yolo_annotations_for_image)

    print("标注转换完成.")
    print(f"总共处理图片: {images_processed}")
    print(f"成功转换的边界框: {successful_conversions}")
    print(f"转换失败/跳过的边界框或图片问题: {conversion_errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将JSON标注转换为YOLO TXT格式。")
    parser.add_argument("--json_file", type=str, required=True, 
                        help="输入的JSON标注文件路径 (例如: data/source_dataset/赛题3.../图片对应输出结果.txt)")
    parser.add_argument("--images_dir", type=str, required=True, 
                        help="包含原始图片的目录路径 (例如: data/source_dataset/赛题3.../)")
    parser.add_argument("--output_labels_dir", type=str, default="../data/processed_yolo_data/labels", 
                        help="输出YOLO TXT标注文件的目录路径 (默认: ../data/processed_yolo_data/labels)")
    parser.add_argument("--class_id", type=int, default=0, help="用于标注的类别ID (默认: 0 for tennis_ball)")

    args = parser.parse_args()

    # 根据脚本位置调整相对路径为绝对路径或更可靠的相对路径
    # 假设脚本在 src/ 目录下，输入的路径可能是相对于项目根目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # 项目根目录 (src的上一级)

    # 解析输入路径，使其相对于项目根目录（如果它们不是绝对路径）
    if not os.path.isabs(args.json_file):
        args.json_file = os.path.join(project_root, args.json_file)
    if not os.path.isabs(args.images_dir):
        args.images_dir = os.path.join(project_root, args.images_dir)
    if not os.path.isabs(args.output_labels_dir):
        args.output_labels_dir = os.path.join(project_root, args.output_labels_dir)

    process_annotations(args.json_file, args.images_dir, args.output_labels_dir, args.class_id) 