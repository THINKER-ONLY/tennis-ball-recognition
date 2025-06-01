# 输出格式中文说明

本项目在进行目标检测后，可以生成多种格式的输出，方便用户查看和进一步处理。

## 1. 可视化输出 (图片/视频)

当使用 `--save_images` (针对图片输入或 `--save_video` (针对视频输入) 参数时，脚本会将处理后的媒体文件保存到磁盘。

- **保存路径**: 
  - 默认情况下，保存在 `results/detection_output_MODELNAME/` 目录下。
    - `MODELNAME` 是所用模型文件名的基本名称 (例如，`yolov8n` 如果使用的是 `yolov8n.pt`)。
  - 可以通过 `--output_image_path` (针对图片) 或 `--output_video_path` (针对视频) 参数自定义保存路径或文件名。
- **内容**: 
  - 输出的图片/视频帧会带有检测到的目标的边界框。
  - 每个边界框旁边通常会显示类别名称和置信度分数。
  - 可以通过 `--hide_labels` 和 `--hide_conf` 参数来隐藏这些信息。

**示例文件名 (默认情况下):**
- 输入 `my_image.jpg` -> 输出 `results/detection_output_yolov8n/my_image.jpg`
- 输入 `my_video.mp4` -> 输出 `results/detection_output_yolov8n/my_video.mp4`

## 2. JSON 输出 (`detected_results.json`)

当使用 `--save_json` 参数时，脚本会将所有检测到的目标的详细信息汇总到一个 JSON 文件中。

- **保存路径**:
  - 默认情况下，文件名为 `detected_results.json`，保存在 `results/detection_output_MODELNAME/` 目录下。
  - 可以通过 `--output_json_path` 参数自定义 JSON 文件的保存路径和名称 (例如 `my_custom_results.json`)。

- **JSON 结构**: 
  该 JSON 文件是一个列表 (list)，列表中的每个元素是一个字典 (dict)，代表一个输入图片中检测到的所有目标。如果输入是单个图片，则列表只包含一个元素。如果输入是一个文件夹或视频，则列表包含多个元素，每个元素对应一个已处理的图片或视频帧。

  每个字典 (代表一张图片或一帧) 的结构如下：

  ```json
  [
    {
      "image_path": "path/to/input/image1.jpg",
      "detections": [
        {
          "class_name": "tennis_ball",
          "class_id": 0,
          "confidence": 0.8567,
          "box_xyxy_original": [100.5, 200.0, 150.8, 280.3],
          "box_xywh_original": [125.65, 240.15, 50.3, 80.3],
          "box_xyxyn_normalized": [0.15625, 0.3125, 0.235625, 0.43796875],
          "box_xywhn_normalized": [0.1959375, 0.375234375, 0.079375, 0.12546875]
        },
        {
          "class_name": "tennis_ball",
          "class_id": 0,
          "confidence": 0.7234,
          "box_xyxy_original": [350.0, 400.0, 380.0, 450.0],
          "box_xywh_original": [365.0, 425.0, 30.0, 50.0],
          "box_xyxyn_normalized": [0.546875, 0.625, 0.59375, 0.703125],
          "box_xywhn_normalized": [0.5703125, 0.6640625, 0.046875, 0.078125]
        }
        // ... 更多检测到的目标
      ]
    },
    {
      "image_path": "path/to/input/image2.png", // 或者 "frame_00123.jpg" (对于视频帧)
      "detections": [
        // ... image2.png 中的检测结果
      ]
    }
    // ... 更多图片/帧的结果
  ]
  ```

- **字段说明**: 
  - `image_path`: 被处理的原始图片的路径。对于视频，这可能是代表帧的临时文件名或帧号。
  - `detections`: 一个列表，包含该图片中所有检测到的目标。每个目标是一个字典。
    - `class_name`: 检测到的目标类别名称 (例如, "tennis_ball")。
    - `class_id`: 目标类别的整数 ID (由模型定义，例如 0)。
    - `confidence`: 模型对该检测结果的置信度分数 (0.0 到 1.0)。
    - `box_xyxy_original`: 边界框在原始图片坐标系下的坐标，格式为 `[x_min, y_min, x_max, y_max]` (左上角和右下角点的像素坐标)。
    - `box_xywh_original`: 边界框在原始图片坐标系下的坐标，格式为 `[x_center, y_center, width, height]` (中心点和框的宽高，像素单位)。
    - `box_xyxyn_normalized`: 归一化的边界框坐标，格式为 `[x_min_norm, y_min_norm, x_max_norm, y_max_norm]`。所有值都在 0.0 到 1.0 之间，通过将原始像素坐标除以图片的宽度或高度得到。
    - `box_xywhn_normalized`: 归一化的边界框坐标，格式为 `[x_center_norm, y_center_norm, width_norm, height_norm]`。所有值都在 0.0 到 1.0 之间。

  如果某张图片没有检测到任何目标 (或者所有检测结果的置信度都低于阈值)，则其对应的 `detections` 列表将为空 `[]`。

## 3. 训练输出

模型训练的输出保存在 `tennis_ball_runs/train/` 或 YOLO 默认的 `runs/train/<experiment_name>/` 目录下。
主要包括：
- **`weights/`**: 
  - `best.pt`: 在验证集上性能最好的模型权重。
  - `last.pt`: 最后一次训练迭代的模型权重。
- **`results.csv`**: 包含每个训练周期 (epoch) 的各种指标 (如损失、mAP 等) 的 CSV 文件。
- **PNG 图表**: 
  - 损失曲线 (例如 `loss.png`, `train_batch*.png`)。
  - mAP 曲线 (例如 `results.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`, `F1_curve.png`)。
  - 混淆矩阵 (`confusion_matrix.png`)。
  - 标签分布图等。
- **`args.yaml`**: 记录了本次训练使用的所有参数。

这些输出对于分析训练过程、评估模型性能以及选择最佳模型至关重要。

## 4. `stdout` (标准输出)

脚本在运行时会在控制台打印日志信息，包括：
- 使用的模型、输入源、参数配置。
- 每张图片/每帧的处理进度 (如果使用 `tqdm` 进度条)。
- 检测到目标的简要信息 (例如，`image1.jpg: 1 tennis_ball`)。
- 保存文件的路径。
- 训练时的 epoch 信息、损失、指标等。

这些实时日志有助于监控脚本的运行状态和快速定位问题。 