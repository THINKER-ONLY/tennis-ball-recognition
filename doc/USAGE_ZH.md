# 项目使用指南

本文档旨在帮助您理解并有效使用本项目。项目提供了两种主要的使用模式，以适应不同的需求：通过 `run.sh` 脚本进行自动化操作，以及直接运行Python脚本进行更精细的控制。

**重要前提**：在执行任何操作之前，请确保您已根据 `SETUP_ZH.md` 的说明，正确创建并激活了Python虚拟环境。
```bash
# 示例: 进入项目根目录并激活环境
cd /path/to/tennis-ball-recognition
source .venv/bin/activate
```

---

## 模式一：使用 `run.sh` (推荐的自动化流程)

这是使用本项目的首选方式，它简化了从模型查找、训练到检测的完整流程。

### 1. 快速上手

1.  **准备输入文件**:
    将您想要检测的图片文件 (例如 `.jpg`) 或视频文件 (例如 `.mp4`) 放入项目根目录下的 `imgs/` 文件夹中。

2.  **运行脚本**:
    在已激活虚拟环境的项目根目录下，执行：
    ```bash
    bash run.sh
    ```

### 2. `run.sh` 详解

`run.sh` 脚本的核心逻辑是：**优先检测，若无模型则自动训练，训练后再次检测。**

**工作流程图**:
`bash run.sh` -> 调用 `src/run_cli.py` (传递参数) -> `run_cli.py` 导入并调用 `src/process.py` 中的 `process_img` 函数

**主要行为**:

1.  **智能模型查找**:
    *   它会按以下顺序智能地查找可用的最佳模型：
        1.  项目根目录下的 `best.pt` (用户手动放置的最高优先级模型)。
        2.  项目根目录下的 `last.pt`。
        3.  `tennis_ball_runs/` 目录下最新一次训练所生成的 `weights/best.pt`。
        4.  `tennis_ball_runs/` 目录下最新一次训练所生成的 `weights/last.pt`。

2.  **执行检测 (如果找到模型)**:
    *   它会调用 `src/run_cli.py` 脚本，并将检测参数（如模型路径、输入源等）传递给它。
    *   **输入源**: 默认配置为 `imgs/` 目录。
    *   **结果输出**: 检测结果将保存在 `results/` 目录下的一个新子文件夹中。
    *   **自定义参数**: 您可以直接编辑 `run.sh` 文件顶部的全局配置区域，来修改默认的输入源 (`INPUT_SOURCE`)、是否显示画面 (`SHOW_OUTPUT`) 或是否保存结果 (`SAVE_OUTPUT`) 等。

3.  **自动训练 (如果未找到可用模型)**:
    *   如果上述查找步骤均未找到任何模型，脚本会自动进入训练模式。
    *   它会调用 `yolo train` 命令，使用 `dataset.yaml` 配置文件进行训练。
    *   训练完成后，它会使用新生成的最佳模型，自动对 `imgs/` 目录进行一次检测。

---

## 模式二：直接运行 Python 脚本 (用于调试与精细控制)

### 1. 运行检测：`src/run_cli.py`

`src/run_cli.py` 是项目的命令行功能入口。当您需要绕过 `run.sh` 的自动查找逻辑，使用特定参数进行检测时，可以直接调用它。

**命令示例**:
```bash
python src/run_cli.py \
    --weights path/to/your/model.pt \
    --source path/to/your/input_source \
    --output_dir results/my_custom_output \
    --conf_thres 0.4 \
    --save_vid
```
**主要参数**:
*   `--weights`: (必需) 指定要使用的模型权重文件。
*   `--source`: (必需) 指定输入源 (图片、视频或目录)。
*   `--output_dir`: 指定输出目录。
*   ...以及其他如 `--conf_thres`, `--iou_thres`, `--show_vid` 等参数。

**运行 `python src/run_cli.py --help` 查看所有可用参数。**

### 2. 快速测试核心API：`src/process.py`

`src/process.py` 本身也可以直接运行，但这仅用于**快速测试**其核心的 `process_img` 函数。

**命令示例**:
```bash
python src/process.py
```

**行为说明**:
*   此命令会执行文件末尾的 `if __name__ == '__main__':` 测试代码。
*   它会**忽略所有命令行参数**。
*   它会使用**硬编码在 `process.py` 内部**的默认模型 (`yolov8n.pt`) 和默认参数。
*   它会处理 `imgs/` 目录下的所有图片，并在终端打印出每个图片的检测结果和耗时。
*   **用途**: 主要用于隔离环境，验证 `process_img` 函数本身是否工作正常，而不受外部参数影响。

### 3. 其他辅助脚本

项目 `src/` 目录下还包含数据处理的辅助脚本，它们也支持命令行调用：
*   `src/convert_annotations.py`: 用于将特定格式的标注文件转换为YOLO TXT格式。
*   `src/split_dataset.py`: 用于将完整的数据集划分为训练集和验证集。

您可以运行 `python src/script_name.py --help` 来查看它们各自的用法和参数。

## 4. 关于输入与输出

*   **检测输入**: 
    *   使用 `run.sh` 时，默认从 `input/` 目录读取。
    *   使用 `src/process.py` 时，通过 `--source` 参数指定。
*   **检测输出**: 
    *   通常保存在 `results/` 目录下的子文件夹中。
    *   详细的输出文件结构（标注图片/视频、JSON数据文件）请参考 `OUTPUT_FORMAT_ZH.md` 或项目根 `README.md` 的"输出结果"部分。
*   **训练输出**: 
    *   保存在 `tennis_ball_runs/` 目录下，每个训练会有一个独立的子文件夹。

更多细节请参考项目根目录的 `README.md`。 