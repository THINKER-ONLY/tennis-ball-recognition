import os
import json
import cv2
import argparse
from tqdm import tqdm # 用于显示进度条

def convert_bbox_to_yolo(img_width: int, img_height: int, 
                           bbox_x: int, bbox_y: int, 
                           bbox_w: int, bbox_h: int, 
                           class_id: int = 0):
    """
    将单个边界框从 (左上角x, 左上角y, 宽, 高) 绝对像素值转换为YOLO格式。

    Args:
        img_width (int): 图像宽度 (单位: 像素)。
        img_height (int): 图像高度 (单位: 像素)。
        bbox_x (int): 边界框左上角 x 坐标 (单位: 像素)。
        bbox_y (int): 边界框左上角 y 坐标 (单位: 像素)。
        bbox_w (int): 边界框宽度 (单位: 像素)。
        bbox_h (int): 边界框高度 (单位: 像素)。
        class_id (int): 类别 ID。默认为 0。

    Returns:
        str or None: YOLO格式的标注字符串 "class_id x_center_norm y_center_norm width_norm height_norm"。
                     如果边界框无效或尺寸为零，则返回 None。
    """
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

    # 确保归一化值在 [0, 1] 范围内
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    if width_norm == 0 or height_norm == 0:
        # print(f"警告: 归一化后宽度/高度为0。bbox_w={bbox_w}, bbox_h={bbox_h}, img_w={img_width}, img_h={img_height}")
        return None

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"

def process_annotations(json_file_path: str, images_dir_path: str, 
                        output_labels_dir_path: str, default_class_id: int = 0):
    """
    读取JSON标注文件，将其内容转换为YOLO格式，并保存到指定的输出目录。

    Args:
        json_file_path (str): 输入的JSON标注文件的路径。
        images_dir_path (str): 包含原始图片的目录路径。
                               用于获取图片尺寸以进行坐标归一化。
        output_labels_dir_path (str): 输出YOLO TXT标注文件的目录路径。
                                    如果目录不存在，将会被创建。
        default_class_id (int): 用于所有标注的类别ID。默认为 0。
    """
    print(f"正在加载JSON标注文件: {json_file_path}")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: JSON文件 '{json_file_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: JSON文件 '{json_file_path}' 格式无效。")
        return
    except Exception as e:
        print(f"读取JSON文件 '{json_file_path}' 时发生未知错误: {e}")
        return

    if not os.path.exists(images_dir_path):
        print(f"错误: 图片目录 '{images_dir_path}' 未找到。")
        return

    os.makedirs(output_labels_dir_path, exist_ok=True)
    print(f"YOLO标注将保存到: {output_labels_dir_path}")

    conversion_errors = 0
    successful_conversions = 0
    images_processed = 0
    images_skipped_missing = 0
    images_skipped_read_error = 0

    print("开始转换标注...")
    for image_filename, bboxes in tqdm(annotations_data.items(), desc="处理图片中"):
        image_path = os.path.join(images_dir_path, image_filename)

        if not os.path.exists(image_path):
            print(f"警告: 图片文件 '{image_filename}' 在 '{images_dir_path}' 中未找到，跳过。")
            images_skipped_missing += 1
            # 如果图片不存在，其所有预期的bbox都算作错误
            conversion_errors += len(bboxes) if isinstance(bboxes, list) else 0 
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: OpenCV无法读取图片 '{image_filename}'，跳过。")
                images_skipped_read_error += 1
                conversion_errors += len(bboxes) if isinstance(bboxes, list) else 0
                continue
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"读取图片 '{image_filename}' 尺寸时出错: {e}，跳过。")
            images_skipped_read_error += 1
            conversion_errors += len(bboxes) if isinstance(bboxes, list) else 0
            continue
        
        images_processed += 1
        yolo_annotations_for_image = []

        if not isinstance(bboxes, list):
            # 对于没有有效bbox列表的图片 (例如json中值为null或非列表)，也创建空label文件
            pass 
        else:
            for bbox_data in bboxes:
                try:
                    x, y, w, h = bbox_data['x'], bbox_data['y'], bbox_data['w'], bbox_data['h']
                    yolo_format_str = convert_bbox_to_yolo(img_width, img_height, x, y, w, h, default_class_id)
                    if yolo_format_str:
                        yolo_annotations_for_image.append(yolo_format_str)
                        successful_conversions +=1
                    else:
                        # convert_bbox_to_yolo 内部会打印无效框的警告
                        conversion_errors += 1
                except KeyError as ke:
                    print(f"警告: 图片 '{image_filename}' 的标注中缺少键: {ke}。标注: {bbox_data}。跳过此框。")
                    conversion_errors += 1
                except Exception as ex_bbox:
                    print(f"警告: 处理图片 '{image_filename}' 的一个边界框时出错: {ex_bbox}。标注: {bbox_data}。跳过此框。")
                    conversion_errors += 1

        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_file_path = os.path.join(output_labels_dir_path, label_filename)

        try:
            with open(label_file_path, 'w', encoding='utf-8') as f_label:
                for line in yolo_annotations_for_image:
                    f_label.write(line + "\n")
        except Exception as e_write:
            print(f"错误: 无法写入标注文件 '{label_file_path}': {e_write}")
            # 之前成功转换的这个图片的标注现在也算失败
            successful_conversions -= len(yolo_annotations_for_image)
            conversion_errors += len(yolo_annotations_for_image)

    print("标注转换完成.")
    print(f"总共检查的图片条目 (来自JSON): {len(annotations_data)}")
    print(f"实际成功处理的图片: {images_processed}")
    print(f"因文件缺失跳过的图片: {images_skipped_missing}")
    print(f"因读取错误跳过的图片: {images_skipped_read_error}")
    print(f"成功转换的边界框: {successful_conversions}")
    print(f"转换失败/跳过的边界框或图片问题导致的计数: {conversion_errors}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将JSON标注转换为YOLO TXT格式。")
    parser.add_argument(
        "--json_file", 
        type=str, 
        required=True, 
        help="输入的JSON标注文件路径 (例如: data/source_dataset/annotations.json)"
    )
    parser.add_argument(
        "--images_dir", 
        type=str, 
        required=True, 
        help="包含原始图片的目录路径 (例如: data/source_dataset/images/)"
    )
    parser.add_argument(
        "--output_labels_dir", 
        type=str, 
        default="../data/processed_yolo_data/labels", 
        help="输出YOLO TXT标注文件的目录路径。如果脚本在 src/ 下运行，此相对路径指向项目根目录下的 data/processed_yolo_data/labels。 (默认值)"
    )
    parser.add_argument(
        "--class_id", 
        type=int, 
        default=0, 
        help="用于标注的类别ID (默认为0, 例如 'tennis_ball')"
    )

    args = parser.parse_args()

    # 确定项目根目录 (假设此脚本位于 src/ 目录下)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    # 将相对路径参数转换为基于项目根目录的绝对路径
    def resolve_path(path_arg):
        if not os.path.isabs(path_arg):
            return os.path.abspath(os.path.join(project_root, path_arg))
        return path_arg

    json_file_resolved = resolve_path(args.json_file)
    images_dir_resolved = resolve_path(args.images_dir)
    output_labels_dir_resolved = resolve_path(args.output_labels_dir)

    print(f"脚本运行目录: {current_script_dir}")
    print(f"项目根目录被假定为: {project_root}")
    print(f"解析后的JSON文件路径: {json_file_resolved}")
    print(f"解析后的图片目录路径: {images_dir_resolved}")
    print(f"解析后的输出标签目录路径: {output_labels_dir_resolved}")

    process_annotations(json_file_resolved, 
                        images_dir_resolved, 
                        output_labels_dir_resolved, 
                        args.class_id) 