import os
import time
import cv2
from ultralytics import YOLO
import argparse
import json

#
#参数:
#   img_path: 要识别的图片的路径
#
#返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 "图片对应输出结果.txt" 中一张图片对应的结果
#
def process_img(img_path):
    pass

#
#以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
#但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
#因此提交时请根据情况删除不必要的额外代码
#
# if __name__=='__main__':
#     imgs_folder = './imgs/'
#     img_paths = os.listdir(imgs_folder)
#     def now():
#         return int(time.time()*1000)
#     last_time = 0
#     count_time = 0
#     max_time = 0
#     min_time = now()
#     for img_path in img_paths:
#         print(img_path,':')
#         last_time = now()
#         result = process_img(imgs_folder+img_path)
#         run_time = now() - last_time
#         print('result:\\n',result)
#         print('run time: ', run_time, 'ms')
#         print()
#         count_time += run_time
#         if run_time > max_time:
#             max_time = run_time
#         if run_time < min_time:
#             min_time = run_time
#     print('\\n')
#     print('avg time: ',int(count_time/len(img_paths)),'ms')
#     print('max time: ',max_time,'ms')
#     print('min time: ',min_time,'ms')

# ---------------------------------------------------------------------------- #
#                                 参数解析器                                  #
# ---------------------------------------------------------------------------- #
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 网球检测脚本")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="模型权重文件的路径，例如 yolov8n.pt 或 best.pt")
    parser.add_argument("--source", type=str, required=True, help="输入源：图片路径、视频路径、摄像头ID (例如 0) 或 RTSP/HTTP 流, 或包含图片的目录")
    parser.add_argument("--output_dir", type=str, default="../video/outputs", help="检测结果输出目录 (用于保存视频或绘制了检测框的图片)")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="目标置信度阈值")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="NMS (非极大值抑制) 的 IOU 阈值")
    parser.add_argument("--show_vid", action="store_true", help="是否实时显示处理后的视频流/图片")
    parser.add_argument("--save_vid", action="store_true", help="是否保存处理后的视频/绘制了检测框的图片")
    parser.add_argument("--save_json", action="store_true", help="是否将检测结果保存为JSON文件")
    parser.add_argument("--output_json_path", type=str, default="../results/detections.json", help="输出JSON文件的路径")
    parser.add_argument("--target_class", type=str, default="tennis ball", help="目标检测类别名称 (例如 'tennis ball')，设为空字符串则检测所有类别")
    return parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                 核心检测逻辑                                 #
# ---------------------------------------------------------------------------- #
def detect_tennis_ball(weights_path, source, output_dir, conf_thres, iou_thres, show_vid, save_vid, save_json, output_json_path, target_class_name):
    """
    使用 YOLOv8 模型进行网球检测。

    参数:
        weights_path (str): 模型权重路径。
        source (str): 输入源 (图片/视频路径, 摄像头ID, 图片目录)。
        output_dir (str): 检测结果的输出目录。
        conf_thres (float): 置信度阈值。
        iou_thres (float): IOU 阈值。
        show_vid (bool): 是否显示视频处理结果。
        save_vid (bool): 是否保存处理后的视频/图片。
        save_json (bool): 是否将结果保存为JSON文件。
        output_json_path (str): JSON文件的输出路径。
        target_class_name (str): 目标检测类别名称 (例如 'tennis ball')，设为空字符串则检测所有类别
    """
    print(f"正在加载模型: {weights_path}...")
    try:
        model = YOLO(weights_path)
        print("模型加载成功.")
        # 获取目标类别的ID (如果指定了目标类别)
        target_class_id = None
        if target_class_name and target_class_name.strip():
            # 确保模型名称与 target_class_name 大小写和空格一致
            # model.names 是一个 {id: name} 的字典
            # 我们需要找到 name 对应的 id
            found_target_id = False
            for class_id, name in model.names.items():
                if name.lower() == target_class_name.strip().lower():
                    target_class_id = class_id
                    found_target_id = True
                    print(f"目标类别: '{target_class_name}' (ID: {target_class_id})")
                    break
            if not found_target_id:
                print(f"警告: 在模型类别中未找到指定的类别 '{target_class_name}'. 将检测所有类别.")
                print(f"可用类别: {model.names}")
        else:
            print("未指定目标类别，将检测所有类别.")

    except Exception as e:
        print(f"模型加载或类别查找失败: {e}")
        return

    print(f"正在对源进行预测: {source}...")
    try:
        # 尝试将 source 转换为整数以判断是否为摄像头 ID
        is_camera = False
        try:
            int_source = int(source)
            is_camera = True
        except ValueError:
            is_camera = False
        
        is_video_file = not is_camera and os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))

        results_generator = model.predict(source, stream=True, conf=conf_thres, iou=iou_thres)
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        return

    all_detections_for_json = {}
    video_writer = None

    if save_vid or save_json:
        if save_json:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        if save_vid:
             os.makedirs(output_dir, exist_ok=True)

    print("开始处理检测结果...")
    first_frame = True
    for i, result in enumerate(results_generator):
        original_frame_for_video = result.orig_img # 获取原始帧用于视频写入
        annotated_frame = result.plot() # YOLO自带的plot功能绘制所有检测结果

        # 如果需要类别过滤，则重新绘制仅包含目标类别的框
        current_frame_detections_json = [] # 用于当前帧的JSON数据
        filtered_boxes_for_log = [] # 用于控制台日志

        if target_class_id is not None:
            # 创建一个新的图像副本用于绘制过滤后的框，避免在原始annotated_frame上重复绘制
            filtered_annotated_frame = original_frame_for_video.copy()
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == target_class_id:
                    # 绘制单个框到 filtered_annotated_frame
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    label = f"{model.names[class_id]} {box.conf[0]:.2f}"
                    cv2.rectangle(filtered_annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(filtered_annotated_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 收集用于JSON和日志的数据
                    xmin, ymin, xmax, ymax = xyxy
                    detection_data = {"x": xmin, "y": ymin, "w": xmax - xmin, "h": ymax - ymin}
                    current_frame_detections_json.append(detection_data)
                    filtered_boxes_for_log.append((model.names[class_id], float(box.conf[0]), xyxy))
            annotated_frame_to_show_save = filtered_annotated_frame # 更新用于显示和保存的帧
        else:
            # 如果不过滤，直接使用YOLO plot的结果，并收集所有框的信息
            annotated_frame_to_show_save = annotated_frame
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                detection_data = {"x": xmin, "y": ymin, "w": xmax - xmin, "h": ymax - ymin}
                current_frame_detections_json.append(detection_data)
                filtered_boxes_for_log.append((model.names[int(box.cls[0])], float(box.conf[0]), [xmin, ymin, xmax, ymax]))

        if show_vid:
            cv2.imshow("YOLOv8 Detection", annotated_frame_to_show_save)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        current_frame_identifier = os.path.basename(result.path) if result.path and not is_video_file and not is_camera else f"frame_{i}"
        if is_video_file and result.path:
             current_frame_identifier = f"{os.path.basename(result.path)}_frame_{i}" # 视频帧加上源文件名

        if save_vid:
            if is_video_file:
                if first_frame:
                    # 获取视频属性以初始化VideoWriter
                    # result.orig_shape 应该是 (height, width)
                    h, w = original_frame_for_video.shape[:2]
                    video_out_path = os.path.join(output_dir, f"processed_{os.path.basename(source)}")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
                    # 尝试从原始视频获取FPS，如果失败则使用默认值 (例如30)
                    # 注意：YOLO的result对象可能不直接提供FPS，需要自行获取或预设
                    fps = 30 # 默认值，可以尝试从输入视频获取
                    # cap = cv2.VideoCapture(source) 
                    # fps = cap.get(cv2.CAP_PROP_FPS) 
                    # cap.release()
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
                    print(f"开始将处理后的视频保存到: {video_out_path}")
                    first_frame = False
                if video_writer:
                    video_writer.write(annotated_frame_to_show_save)
            else: # 如果是图片或摄像头帧
                output_image_path = os.path.join(output_dir, f"result_{current_frame_identifier}.jpg")
                cv2.imwrite(output_image_path, annotated_frame_to_show_save)
                # print(f"已保存带标注的图片: {output_image_path}") # 可以取消注释以获得更详细日志

        if save_json and current_frame_detections_json: # 只保存包含目标类别的检测结果
            # 对于图片目录，result.path是唯一的图片名。对于视频/摄像头，使用帧标识符。
            json_key = os.path.basename(result.path) if result.path and os.path.isfile(result.path) else current_frame_identifier
            all_detections_for_json[json_key] = current_frame_detections_json

        print(f"--- {current_frame_identifier} ---")
        if filtered_boxes_for_log:
            for class_name, confidence, bbox_xyxy in filtered_boxes_for_log:
                print(f"  检测到: {class_name}, 置信度: {confidence:.2f}, 边界框 (xyxy): {bbox_xyxy}")
        else:
            print("  未检测到目标类别.")

    if video_writer:
        video_writer.release()
        print("处理后的视频已保存.")

    if save_json and all_detections_for_json:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_detections_for_json, f, ensure_ascii=False, indent=4)
            print(f"检测结果已保存到JSON文件: {output_json_path}")
        except Exception as e:
            print(f"保存JSON文件失败: {e}")

    if show_vid:
        cv2.destroyAllWindows()
    print("检测处理完毕.")

# ---------------------------------------------------------------------------- #
#                                   主函数                                    #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()
    detect_tennis_ball(
        weights_path=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        show_vid=args.show_vid,
        save_vid=args.save_vid,
        save_json=args.save_json,
        output_json_path=args.output_json_path,
        target_class_name=args.target_class
    )
    print("网球检测流程结束.")
