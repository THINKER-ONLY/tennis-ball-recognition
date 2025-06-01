# 项目使用中文说明

本文档介绍如何使用本项目进行网球检测和模型训练，主要通过 `run.sh` 脚本进行操作。

## 1. `run.sh` 脚本概览

`run.sh` 是项目的主要入口点，它自动化了环境检查、模型查找、训练和检测流程。它支持多种操作模式，通过命令行参数进行配置。

### 1.1 核心功能

- **环境检查**: 自动检查 Python 虚拟环境 (`.venv`) 是否存在并激活。
- **模型查找/训练**: 
  - 优先查找项目根目录下的 `.pt` 模型文件（如 `yolov8n.pt`, `yolo11n.pt`）。
  - 如果找不到，则查找 `tennis_ball_runs/train/weights/best.pt`（即上次训练的最佳模型）。
  - 如果仍然找不到，并且设置了 `--train_if_no_model`，则会自动使用 `data/dataset.yaml` 进行模型训练。
- **目标检测**: 使用找到的或新训练的模型对指定的输入源进行检测。

### 1.2 `run.sh` 常用参数

执行 `./run.sh --help` 可以查看所有可用参数及其说明。以下是一些关键参数：

- **操作模式 (必须指定一个):**
  - `--run_detection`: 运行目标检测。
  - `--train_model`: 强制进行模型训练 (即使已存在模型)。

- **输入源 (当 `--run_detection` 时必须指定一个):**
  - `--input_image <path_to_image>`: 指定单张输入图片。
  - `--input_video <path_to_video>`: 指定输入视频文件。
  - `--input_folder <path_to_folder>`: 指定包含多张图片的输入文件夹。

- **检测相关参数 (当 `--run_detection` 时可选):**
  - `--target_class <class_name>`: 要检测的目标类别名称 (例如: `tennis_ball`)。默认为 `tennis_ball`。
  - `--confidence_threshold <float>`: 置信度阈值 (0.0 到 1.0)。默认为 0.25。
  - `--iou_threshold <float>`: IOU (Intersection over Union) 阈值，用于NMS (非极大值抑制)。默认为 0.7。
  - `--save_images`: 保存处理后带有检测框的图片。
  - `--save_video`: 保存处理后带有检测框的视频 (如果输入是视频)。
  - `--save_json`: 将检测结果以 JSON 格式保存到文件 (`results/detection_output_MODELNAME/detected_results.json`)。
  - `--output_image_path <path>`: 指定处理后图片的保存路径 (如果 `--save_images` 启用)。
  - `--output_video_path <path>`: 指定处理后视频的保存路径 (如果 `--save_video` 启用)。
  - `--output_json_path <path>`: 指定 JSON 结果的保存路径 (如果 `--save_json` 启用)。
  - `--hide_labels`: 在输出图像/视频中隐藏标签。
  - `--hide_conf`: 在输出图像/视频中隐藏置信度分数。
  - `--show_output`: 实时显示处理后的图像/视频窗口。

- **训练相关参数 (当 `--train_model` 或 `--train_if_no_model` 时可选):**
  - `--epochs <int>`: 训练轮数。默认为 100。
  - `--batch_size <int>`: 批处理大小。默认为 16。
  - `--img_size <int>`: 训练图像尺寸。默认为 640。
  - `--model_to_train <model_name.pt>`: 指定用于训练的基础模型 (例如 `yolov8n.pt`)。如果未提供，脚本会尝试使用默认的或上次训练的模型。
  - `--dataset_yaml <path_to_yaml>`: 指定数据集配置文件。默认为 `data/dataset.yaml`。

- **其他参数:**
  - `--train_if_no_model`: 如果没有找到预训练模型，则自动开始训练。这是 `run.sh` 的默认行为之一。
  - `--custom_model_path <path_to_model.pt>`: 明确指定要使用的模型路径，覆盖自动查找逻辑。
  - `--skip_venv_check`: 跳过虚拟环境检查 (不推荐)。

### 1.3 `run.sh` 示例用法

**前提: 确保已按照 `SETUP_ZH.md` 完成环境设置并激活了虚拟环境。**

1.  **使用默认模型检测单张图片，并保存结果图片和JSON:**
    ```bash
    ./run.sh --run_detection --input_image ./data/source_dataset/some_image.jpg --save_images --save_json
    ```

2.  **使用指定模型检测视频，实时显示并保存处理后的视频:**
    ```bash
    ./run.sh --run_detection --input_video ./video/my_tennis_match.mp4 --custom_model_path yolov8n.pt --show_output --save_video
    ```

3.  **检测图片文件夹中的所有图片，并将结果保存到指定目录:**
    ```bash
    ./run.sh --run_detection --input_folder ./data/source_dataset/test_images/ --save_images --output_image_path ./results/my_custom_output/
    ```

4.  **强制重新训练模型，训练50轮:**
    ```bash
    ./run.sh --train_model --epochs 50
    ```

5.  **如果找不到模型则训练，否则用找到的模型检测图片，并指定目标类别为 'ball':**
    ```bash
    ./run.sh --run_detection --input_image image.png --train_if_no_model --target_class "ball"
    ```
    *(注意: 如果 `target_class` 包含空格或特殊字符，建议使用引号)*

## 2. 直接使用 Python 脚本

虽然推荐使用 `run.sh`，但您也可以直接运行 `src/` 目录下的 Python 脚本。

### 2.1 `src/process.py`

这是核心的检测脚本。它接受与 `run.sh` 中检测部分类似的参数。

```bash
# 激活虚拟环境后
python src/process.py --input_image ./data/source_dataset/some_image.jpg --model_path yolov8n.pt --save_images
```
执行 `python src/process.py --help` 查看其所有参数。

### 2.2 `src/split_dataset.py`

用于划分数据集。
```bash
python src/split_dataset.py --source_dir ./data/source_dataset/all_data --output_dir ./data/processed_yolo_data --split_ratio 0.8 0.1 0.1
```
参数说明:
- `--source_dir`: 包含 `images` 和 `labels` (YOLO TXT格式) 的源数据文件夹。
- `--output_dir`: 输出划分后数据的目标文件夹，会在此创建 `images/{train,val,test}` 和 `labels/{train,val,test}`。
- `--split_ratio`: 训练、验证、测试集的划分比例 (例如 0.8 0.1 0.1 表示 80% 训练, 10% 验证, 10% 测试)。测试集是可选的。
- `--copy_files`: 是否复制文件，默认为是。如果为否，则创建符号链接 (symlinks)。

### 2.3 `src/convert_annotations.py`

用于将其他格式的标注 (例如 COCO JSON) 转换为 YOLO TXT 格式。
假设您的 JSON 文件名为 `annotations.json`，图片在 `images/` 目录：
```bash
python src/convert_annotations.py --json_path annotations.json --output_dir ./yolo_labels/ --image_dir ./images/
```
参数说明:
- `--json_path`: 输入的 COCO 格式 JSON 标注文件路径。
- `--output_dir`: 输出 YOLO TXT 标签文件的目录。
- `--image_dir`: (可选) 包含与标注对应图片的目录，用于获取图片尺寸信息。如果提供，脚本会检查图片是否存在。
- `--class_mapping`: (可选) JSON 格式的字符串，用于指定类别名称到类别ID的映射，例如 `'{"tennis_ball": 0, "net": 1}'`。如果未提供，将基于JSON中的类别自动生成映射。

## 3. 查看输出

- **检测结果图片/视频**: 默认保存在 `results/detection_output_MODELNAME/` 目录下 (如果启用了保存选项)。
- **JSON 结果**: 默认保存在 `results/detection_output_MODELNAME/detected_results.json` (如果启用了 `--save_json`)。
- **训练输出**: 保存在 `tennis_ball_runs/train/` 目录下，包括模型权重 (`weights/best.pt`, `weights/last.pt`)、日志、图表等。

详细的输出格式请参考 `OUTPUT_FORMAT_ZH.md`。 