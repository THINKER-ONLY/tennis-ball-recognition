# Tennis Ball Recognition (智能捡网球机器人识别)

本项目旨在通过YOLOv8实现对网球的实时目标检测，适用于智能捡网球机器人等场景。

## 项目结构

```
tennis-ball-recognition/
├── .git/                     # Git 版本控制元数据
├── .gitignore                # 指定Git忽略的文件和目录
├── .venv/                    # Python 虚拟环境 (使用 uv 创建)
├── base_models/              # 存放基础预训练模型 (如 yolov8n.pt)
├── data/
│   ├── processed_yolo_data/  # YOLO 格式的已处理数据集
│   │   ├── all_labels/       # 所有图片的YOLO TXT标签
│   │   ├── images/           # 所有图片 (待划分或已划分)
│   │   │   ├── train/        # 训练集图片
│   │   │   └── val/          # 验证集图片
│   │   └── labels/           # YOLO TXT 标签
│   │       ├── train/        # 训练集标签
│   │       └── val/          # 验证集标签
│   └── source_dataset/
│       └── original_competition_data/ # 原始比赛数据集 (图片和result.txt)
├── dataset.yaml              # YOLOv8 数据集配置文件
├── doc/                      # 项目相关文档
├── models/                   # 存放训练好的、可部署的模型 (如 best.pt)
├── README.md                 # 项目说明文件 (本文档)
├── requirements.txt          # 项目Python依赖
├── results/                  # (Git忽略) 存放预测输出的JSON文件
├── run.sh                    # 项目主要执行脚本 (包含环境设置、数据处理、训练、预测等)
├── slide/                    # 项目演示幻灯片
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── convert_annotations.py # 脚本: 原始标注 (JSON) 转 YOLO TXT
│   ├── process.py             # 核心脚本: 目标检测 (图片/视频), 结果输出
│   └── split_dataset.py       # 脚本: 划分数据集为训练/验证集
├── tennis_ball_runs/         # (Git忽略) YOLOv8 训练输出 (权重, 日志, 图表)
└── video/                    # 存放原始视频文件
    └── outputs/              # (Git忽略) 存放处理后的视频文件
```

## 环境搭建

1.  **克隆仓库**:
    ```bash
    git clone <repository-url>
    cd tennis-ball-recognition
    ```

2.  **创建并激活虚拟环境 (使用 uv)**:
    确保已安装 `uv` (Python包管理工具)。
    ```bash
    uv venv
    source .venv/bin/activate
    ```
    如果 `uv` 命令不可用, 请先[安装uv](https://github.com/astral-sh/uv#installation)。

3.  **安装依赖**:
    ```bash
    uv pip install -r requirements.txt
    ```
    如果在安装 `ultralytics` 及其依赖 (如 `nvidia-cublas-cu12`) 时遇到下载超时, 可以尝试增加超时时间:
    ```bash
    export UV_HTTP_TIMEOUT=300 # 或者更大的值
    uv pip install -r requirements.txt
    ```

## 数据准备

1.  **放置原始数据**:
    将比赛提供的图片和 `result.txt` (JSON格式标注文件) 放入 `data/source_dataset/original_competition_data/` 目录。

2.  **转换标注格式**:
    运行脚本将原始JSON标注转换为YOLO TXT格式。
    ```bash
    python src/convert_annotations.py \
        --json_file data/source_dataset/original_competition_data/result.txt \
        --images_dir data/source_dataset/original_competition_data/ \
        --output_labels_dir data/processed_yolo_data/all_labels/ \
        --class_id 0 # 假设网球的类别ID为0
    ```
    确保 `result.txt` 中的JSON格式正确 (键值对间有逗号)。

3.  **复制图片并划分数据集**:
    首先将所有图片复制到统一处理目录:
    ```bash
    cp data/source_dataset/original_competition_data/*.jpg data/processed_yolo_data/images/
    ```
    然后运行脚本划分训练集和验证集 (默认80/20比例):
    ```bash
    python src/split_dataset.py \
        --images_dir data/processed_yolo_data/images/ \
        --labels_dir data/processed_yolo_data/all_labels/ \
        --output_base_dir data/processed_yolo_data/
    ```
    划分后的图片和标签会分别存放在 `data/processed_yolo_data/images/train/`, `data/processed_yolo_data/images/val/`, `data/processed_yolo_data/labels/train/`, `data/processed_yolo_data/labels/val/`。

4.  **创建 `dataset.yaml`**:
    在项目根目录创建 `dataset.yaml` 文件，内容如下:
    ```yaml
    path: data/processed_yolo_data  # 数据集根路径 (相对于此yaml文件)
    train: images/train             # 训练集图片目录 (相对于path)
    val: images/val                 # 验证集图片目录 (相对于path)

    names:
      0: tennis_ball                # 类别名称
    ```

## 模型训练

使用 `yolo` 命令行工具进行训练。建议使用 `yolov8n.pt` 作为预训练模型。
```bash
yolo train \
    model=yolov8n.pt \
    data=dataset.yaml \
    epochs=50 \
    imgsz=640 \
    batch=8 \
    project=tennis_ball_runs \
    name=your_train_name 
    # plots=False # 如果遇到字体下载问题或不需要绘图，可添加此参数
```
训练结果将保存在 `tennis_ball_runs/your_train_name/` 目录下，最佳权重为 `weights/best.pt`。

## 模型推理与结果生成

使用 `src/process.py` 脚本对图片或视频进行检测，并生成符合比赛要求的JSON结果。

**示例 (检测图片文件夹):**
```bash
python src/process.py \
    --weights tennis_ball_runs/your_train_name/weights/best.pt \
    --source data/source_dataset/original_competition_data/ \
    --save_json \
    --output_json_path results/predictions.json \
    --conf_thres 0.4
```

**示例 (检测单个视频并保存带标注的视频):**
```bash
python src/process.py \
    --weights tennis_ball_runs/your_train_name/weights/best.pt \
    --source path/to/your/video.mp4 \
    --save_vid \
    --output_video_path video/outputs/processed_video.mp4 \
    --save_json \
    --output_json_path results/video_predictions.json
```

## 使用 `run.sh` 脚本

项目提供了一个 `run.sh` 脚本，封装了常用的操作流程，包括环境设置、数据预处理、训练和预测等。请查看脚本内的注释以了解详细用法和可配置参数。

**基本用法:**
```bash
bash run.sh --task <task_name> [other_options]
```
例如，执行完整流程 (数据准备、训练、预测):
```bash
bash run.sh --task all \
    --source_data_dir data/source_dataset/original_competition_data \
    --yolo_model yolov8n.pt \
    --epochs 50 \
    --batch_size 8
```

## 注意事项
*   确保所有脚本均在已激活虚拟环境的项目根目录下执行。
*   根据实际硬件资源调整训练时的 `batch` 大小和 `imgsz`。
*   对于非常小的数据集，模型的泛化能力可能有限，建议尽可能扩充数据集。