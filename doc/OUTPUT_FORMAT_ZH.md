# 输出格式说明

本文档描述了在使用本网球目标检测项目后，您可以预期的各种输出文件的格式和位置。

## 1. 检测结果：可视化输出 (图片/视频)

当通过 `run.sh` (内部配置 `SAVE_OUTPUT="true"`) 或直接运行 `src/process.py` (使用 `--save_vid` 参数) 并启用保存时，脚本会保存带有标注的图片或视频。

*   **内容**: 输出的图片或视频帧上会绘制检测到的目标边界框。每个边界框旁边通常会显示类别名称和置信度分数 (这些可以通过 `src/process.py` 的参数或内部逻辑调整，例如 `--hide_labels`, `--hide_conf` 在 `yolo predict` 时可用，但 `src/process.py` 目前是自行绘制的)。
*   **保存路径**:
    *   **使用 `run.sh`**:
        *   结果保存在 `results/detection_output_[model_name]/` 目录下 (`[model_name]` 是所用模型的文件名，如 `best` 或 `yolo11n`)。
        *   如果输入源 (`INPUT_SOURCE` 在 `run.sh` 中设置) 是一个目录 (例如 `input/`)，包含 `image1.jpg`：
            *   标注图片可能保存为：`results/detection_output_[model_name]/image1/result_image1.jpg`
        *   如果输入源是单个视频 `my_video.mp4`：
            *   标注视频可能保存为：`results/detection_output_[model_name]/processed_my_video.mp4`
    *   **直接运行 `src/process.py`**:
        *   结果保存在 `--output_dir` 指定的目录下。
        *   如果 `--source` 是一个目录，例如 `my_images/`，其中包含 `image1.jpg`：
            *   标注图片保存为：`[output_dir]/image1/result_image1.jpg`
        *   如果 `--source` 是单个图片文件 `image1.jpg`：
            *   标注图片保存为：`[output_dir]/result_image1.jpg` (或者基于 `process.py` 内部逻辑，也可能是 `[output_dir]/image1/result_image1.jpg`)
        *   如果 `--source` 是单个视频文件 `my_video.mp4`：
            *   标注视频保存为：`[output_dir]/processed_my_video.mp4`

## 2. 检测结果：JSON 数据输出

当通过 `run.sh` (内部配置 `SAVE_JSON_OUTPUT="true"`) 或直接运行 `src/process.py` (使用 `--save_json` 参数) 并启用保存时，脚本会将检测到的目标的详细信息以JSON格式保存到文本文件中。

*   **核心机制**：`src/process.py` 会为**每个成功处理的输入图片或视频文件**生成一个对应的 `.txt` 文件，该文件的内容是一个JSON数组，记录了在该图片/视频中检测到的所有目标对象。

*   **文件命名与路径**:
    *   **使用 `run.sh`** (假设 `INPUT_SOURCE` 指向 `input/` 目录):
        *   如果检测了 `input/image1.jpg`，对应的JSON数据文件路径通常是：
          `results/detection_output_[model_name]/image1/image1.txt`
        *   如果检测了 `input/subdir/image2.png`，路径可能是：
          `results/detection_output_[model_name]/subdir_image2/subdir_image2.txt` (具体取决于 `src/process.py` 如何从路径生成 `item_stem`)
        *   如果检测了单个视频 `input/my_video.mp4`，JSON文件路径可能是：
          `results/detection_output_[model_name]/my_video/my_video.txt` (或者直接在 `results/detection_output_[model_name]/my_video.txt`，取决于 `process.py` 对视频的处理方式和 `item_stem` 的生成逻辑)。
          **注意**: `run.sh` 脚本中尝试指定的 `OUTPUT_JSON_PATH_DETECT="${OUTPUT_DIR_DETECT}/result.txt"` 路径，在 `src/process.py` 中由于 `--output_json_path` 参数已弃用，因此**不会**生成这样一个单一的聚合 `result.txt` 文件。而是会按上述方式为每个源图片生成单独的 `.txt` 文件。
    *   **直接运行 `src/process.py`**:
        *   如果 `--source` 是一个目录 (例如 `my_input_images/`)，其中包含 `image1.jpg`：
            *   JSON 文件路径：`[output_dir]/image1/image1.txt`
        *   如果 `--source` 是单个图片文件 `image1.jpg`：
            *   JSON 文件路径：`[output_dir]/image1.txt` (也可能是 `[output_dir]/image1/image1.txt`，取决于 `process.py` 的 `item_stem` 逻辑)
        *   如果 `--source` 是单个视频文件 `my_video.mp4`：
            *   JSON 文件路径： `[output_dir]/my_video.txt` (或 `[output_dir]/my_video/my_video.txt`)

*   **JSON 文件内容结构**:
    每个生成的 `.txt` 文件包含一个JSON数组。数组中的每个元素是一个JSON对象，代表在该图片/视频中检测到的一个目标。示例如下：

    ```json
    [
      {
        "label": "tennis_ball",
        "confidence": 0.9234,
        "bbox": [120, 235, 55, 60]
      },
      {
        "label": "tennis_ball",
        "confidence": 0.8750,
        "bbox": [350, 300, 58, 59]
      }
      // ... 如果图片中有更多网球，则此处有更多对象
    ]
    ```
    如果某张图片没有检测到任何符合条件的目标，则对应的 `.txt` 文件内容会是一个空数组 `[]`。

*   **JSON 对象字段说明**: 
    *   `label` (string): 检测到的目标类别名称 (例如, `tennis_ball`)。这对应于模型训练时在 `dataset.yaml` 中定义的 `names`。
    *   `confidence` (float): 模型对该检测结果的置信度分数，值域通常为 0.0 到 1.0。
    *   `bbox` (array of integers): 边界框在原始图片坐标系下的位置和尺寸，格式为 `[x_min, y_min, width, height]`。
        *   `x_min`: 边界框左上角顶点的 x 坐标 (像素值)。
        *   `y_min`: 边界框左上角顶点的 y 坐标 (像素值)。
        *   `width`: 边界框的宽度 (像素值)。
        *   `height`: 边界框的高度 (像素值)。

## 3. 训练输出

当您运行模型训练时 (无论是通过 `run.sh` 自动触发还是直接使用 `yolo train` 命令)，训练过程的输出会保存在特定的目录中。

*   **保存路径**: 
    *   默认情况下，如果通过 `run.sh` 或直接使用 `yolo train project=tennis_ball_runs name=[your_run_name]`，训练输出会保存在 `tennis_ball_runs/[your_run_name]/` 目录下。
    *   例如：`tennis_ball_runs/first_train/` 或 `tennis_ball_runs/my_custom_run/`。

*   **主要内容**: 每个训练运行的子目录中通常包含：
    *   `weights/` 目录:
        *   `best.pt`: 在验证集上表现最佳的模型权重文件。这是通常用于后续推理的模型。
        *   `last.pt`: 完成所有训练轮次 (epochs) 后最后一次保存的模型权重文件。
    *   `results.csv`: 一个 CSV 文件，记录了每个训练周期的详细指标，包括损失 (loss)、精确率 (precision)、召回率 (recall)、mAP50 (mean Average Precision @ IoU=0.5)、mAP50-95 (mean Average Precision @ IoU=0.5:0.95) 等。
    *   各种图表 (PNG 图片文件，如果训练时启用了 `plots=True` 或默认启用):
        *   `results.png` 或类似名称: 包含训练和验证损失曲线、mAP 曲线等的汇总图表。
        *   `confusion_matrix.png`: 混淆矩阵图，显示了类别之间的分类情况。
        *   `P_curve.png`, `R_curve.png`, `PR_curve.png`, `F1_curve.png`: 精确率、召回率、PR曲线和F1分数曲线图。
        *   其他可能的图表，如标签分布、数据增强效果预览等。
    *   `args.yaml`: 一个YAML文件，记录了本次训练所使用的所有超参数和配置信息。
    *   日志文件 (例如 `events.out.tfevents.*`): TensorBoard 事件文件，可用于更详细地可视化训练过程。

这些训练输出对于监控训练状态、评估模型性能、进行模型比较以及选择最佳模型进行部署至关重要。

## 4. 控制台标准输出 (`stdout`)

脚本在运行时会在控制台（终端）打印大量的实时日志信息，这对于监控和调试非常有帮助：
*   **启动信息**: 使用的模型路径、输入源、关键参数配置等。
*   **进度条**: 在处理多个文件或视频帧时，通常会显示进度条 (例如使用 `tqdm` 库)。
*   **检测信息**: 对于每个检测到的目标，可能会简要打印类别和置信度。
*   **文件保存信息**: 当文件被保存时，通常会打印出保存路径。
*   **训练信息**: 在训练期间，每个 epoch 的开始和结束、当前的损失值、评估指标等会实时更新显示。
*   **错误和警告**: 如果发生任何问题，错误信息和警告也会输出到控制台。

仔细观察控制台输出是理解脚本执行情况和快速定位问题的有效方法。 