# 环境设置与安装指南

本项目依赖 Python 和一些特定的库来运行。强烈建议在 Python 虚拟环境中管理这些依赖，以避免与系统或其他项目的库产生冲突。

## 1. 前提条件

*   **Python**: 请确保您已安装 Python 3.8 或更高版本。
*   **Git**: 需要使用 Git 来克隆项目（如果尚未获取项目代码）。

## 2. 获取项目代码 (如果需要)

如果您尚未下载项目，请使用 Git 克隆。如果您已拥有项目文件夹，请直接进入该目录。
```bash
# 示例: git clone https://github.com/your-username/tennis-ball-recognition.git
cd tennis-ball-recognition
```
**后续所有命令都假定您位于项目根目录 (`tennis-ball-recognition/`) 下执行。**

## 3. 创建并激活Python虚拟环境

项目脚本（如 `run.sh`）会默认检查并使用项目根目录下的 `.venv` 文件夹。

1.  **创建虚拟环境** (使用 Python 内置的 `venv`)
    ```bash
    # 确保您在项目根目录下
    python3 -m venv .venv
    ```

2.  **激活虚拟环境**
    ```bash
    # 确保您在项目根目录下
    source .venv/bin/activate
    ```
    激活成功后，您的命令行提示符前通常会显示 `(.venv)`。
    **重要提示**: 在执行本项目中的任何 Python 脚本或 `run.sh` 脚本之前，**务必先激活虚拟环境**。

## 4. 安装依赖

项目所需的 Python 依赖项已全部记录在根目录的 `requirements.txt` 文件中。该文件经过特殊配置，能够同时支持纯CPU环境和NVIDIA GPU环境。

**执行安装:**
```bash
# 确保虚拟环境已激活
pip install -r requirements.txt
```

**依赖如何工作**:
*   **对于纯CPU用户**: 安装脚本将自动为您下载和安装CPU版本的 `PyTorch`。`requirements.txt` 文件顶部的 `--extra-index-url https://download.pytorch.org/whl/cpu` 指令会确保这一点。
*   **对于NVIDIA GPU用户**: 如果您的系统中已正确安装NVIDIA驱动和CUDA工具包，`pip` 会忽略上述CPU指令，并自动为您安装与CUDA版本匹配的、性能更强的GPU版本 `PyTorch`。

## 5. 准备工作

完成以上步骤后，您的开发环境就准备好了。
*   关于如何**准备输入图片进行检测**，或如何**准备数据集进行训练**，请参考项目根目录下主 `README.md` 文件中的详细指南。

现在，您可以开始使用本项目了！ 