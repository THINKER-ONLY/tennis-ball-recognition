#!/bin/bash

# 这是一个启动脚本示例
# 它可以用来执行数据预处理、模型训练、目标检测等任务

echo "开始执行网球检测项目..."

# --- 虚拟环境管理 (推荐使用 uv) ---
echo "检查/激活 Python 虚拟环境..."

# 检查 .venv 虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "未找到 .venv 虚拟环境。"
    echo "如果您已安装 uv (https://github.com/astral-sh/uv)，可以尝试使用以下命令创建:"
    echo "  uv venv"
    echo "如果您使用的是标准的 venv，可以尝试:"
    echo "  python3 -m venv .venv"
    echo "创建环境后，请先安装依赖。"
fi

# 尝试激活 .venv (如果存在)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "已激活 .venv 虚拟环境."
    echo "确保已安装依赖。如果使用 uv: uv pip install ultralytics opencv-python"
    echo "如果使用 pip: pip install ultralytics opencv-python"
else
    echo "警告: .venv 虚拟环境未激活。请确保已创建并激活虚拟环境，并安装了必要的依赖。"
    echo "依赖包括: ultralytics, opencv-python"
fi
# -------------------------------------


# 运行处理脚本 (示例)
# 请根据您的实际需求修改以下参数：
# --weights: 模型权重路径 (例如 yolov8n.pt, 或您训练的 best.pt)
# --source: 输入源 (图片路径, 视频路径, 摄像头ID, 或包含图片的目录)
# --output_dir: 保存带标注的图片或视频的目录 (如果使用 --save_vid)
# --save_vid: 启用此选项以保存处理后的视频或带标注的图片
# --show_vid: 启用此选项以实时显示处理结果
# --save_json: 启用此选项以将检测结果保存为JSON文件
# --output_json_path: JSON文件的输出路径 (如果使用 --save_json)
# --target_class: 要检测的目标类别 (例如 "tennis ball"，为空则检测所有)

echo ""
echo "运行检测脚本示例 (请根据需要取消注释并修改参数):"

# 示例1: 处理单张图片，显示并保存结果 (图片和JSON)
# echo "运行示例1: 处理单张图片..."
# python src/process.py \
#    --weights yolov8n.pt \
#    --source data/赛题3\ -\ 智能捡网球机器人识别\ -\ 测试图片及结果/1747468171.jpg \
#    --output_dir video/outputs/ \
#    --target_class "tennis ball" \
#    --show_vid \
#    --save_vid \
#    --save_json \
#    --output_json_path results/detected_output.json

# 示例2: 处理整个图片目录，不实时显示，保存标注图片和JSON结果
# echo "运行示例2: 处理图片目录..."
# python src/process.py \
#    --weights yolov8n.pt \
#    --source data/赛题3\ -\ 智能捡网球机器人识别\ -\ 测试图片及结果/ \
#    --output_dir results/annotated_images/ \
#    --target_class "tennis ball" \
#    --save_vid \
#    --save_json \
#    --output_json_path results/batch_detections.json


# 示例3: 处理视频文件，显示处理过程，并保存输出视频和JSON结果
# echo "运行示例3: 处理视频文件..."
# python src/process.py \
#    --weights yolov8n.pt \
#    --source video/sample_video.mp4 \
#    --output_dir video/outputs/ \
#    --target_class "tennis ball" \
#    --show_vid \
#    --save_vid \
#    --save_json \
#    --output_json_path results/video_detections.json

echo ""
echo "脚本执行完毕." 