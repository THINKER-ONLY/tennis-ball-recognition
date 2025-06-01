#!/bin/bash

# 这是一个启动脚本示例
# 它可以用来执行数据预处理、模型训练、目标检测等任务

echo "开始执行网球检测项目..."

# --- 虚拟环境管理 ---
echo "检查/激活 Python 虚拟环境..."
if [ ! -d ".venv" ]; then
    echo "未找到 .venv 虚拟环境。"
    echo "如果您已安装 uv (https://github.com/astral-sh/uv)，可以尝试使用以下命令创建:"
    echo "  uv venv"
    echo "创建环境后，请先安装依赖: uv pip install ultralytics opencv-python tqdm"
    echo "如果您使用的是标准的 venv，可以尝试:"
    echo "  python3 -m venv .venv"
    echo "创建环境后，请先激活并安装依赖: pip install ultralytics opencv-python tqdm"
    exit 1
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "已激活 .venv 虚拟环境."
else
    echo "警告: .venv 虚拟环境未激活。请确保已创建并激活虚拟环境，并安装了必要的依赖。"
    echo "依赖包括: ultralytics, opencv-python, tqdm"
    exit 1
fi
# -------------------------------------

# --- 主要逻辑：智能查找模型，然后训练或检测 ---
TRAIN_OUTPUT_PROJECT_DIR="tennis_ball_runs"
TRAIN_BASE_NAME="first_train"
LATEST_TRAIN_DIR=""
MODEL_TO_USE=""

echo ""
echo "正在查找最新的训练成果..."

# 优先：检查项目根目录下的模型
if [ -f "best.pt" ]; then
    MODEL_TO_USE="best.pt"
    echo "在项目根目录找到最佳模型: ${MODEL_TO_USE}"
elif [ -f "last.pt" ]; then
    MODEL_TO_USE="last.pt"
    echo "在项目根目录找到最后的模型: ${MODEL_TO_USE} (未找到 best.pt)"
fi

# 其次：如果根目录没有，则检查 tennis_ball_runs/ 目录
if [ -z "${MODEL_TO_USE}" ]; then
    CANDIDATE_DIRS=$(ls -d ${TRAIN_OUTPUT_PROJECT_DIR}/${TRAIN_BASE_NAME}* 2>/dev/null | sort -V | tail -n 1)
    if [ -d "${CANDIDATE_DIRS}" ]; then
        LATEST_TRAIN_DIR="${CANDIDATE_DIRS}"
        echo "在 ${TRAIN_OUTPUT_PROJECT_DIR} 中找到最新的训练目录: ${LATEST_TRAIN_DIR}"

        BEST_PT_PATH="${LATEST_TRAIN_DIR}/weights/best.pt"
        LAST_PT_PATH="${LATEST_TRAIN_DIR}/weights/last.pt"

        if [ -f "${BEST_PT_PATH}" ]; then
            MODEL_TO_USE="${BEST_PT_PATH}"
            echo "找到最佳模型: ${MODEL_TO_USE}"
        elif [ -f "${LAST_PT_PATH}" ]; then
            MODEL_TO_USE="${LAST_PT_PATH}"
            echo "未找到 best.pt，将使用最后的模型: ${MODEL_TO_USE}"
        else
            echo "在最新的训练目录 ${LATEST_TRAIN_DIR} 中未找到 best.pt 或 last.pt。"
        fi
    else
        echo "未找到任何 ${TRAIN_BASE_NAME}* 相关的训练目录。"
    fi
fi

if [ ! -z "${MODEL_TO_USE}" ]; then
    echo ""
    echo "将使用模型 ${MODEL_TO_USE} 进行目标检测。"
    echo ""

    # --- 配置检测参数 (使用找到的模型) ---
    MODEL_WEIGHTS="${MODEL_TO_USE}"
    # 你可以按需修改这里的 INPUT_SOURCE
    INPUT_SOURCE="input/2.jpg" 
    # INPUT_SOURCE="data/processed_yolo_data/images/val/" # 验证集图片目录示例
    OUTPUT_DIR="results/detection_output_$(basename ${MODEL_WEIGHTS} .pt)" # 按模型名创建独立输出目录
    TARGET_CLASS="tennis_ball"
    SHOW_OUTPUT="true"
    SAVE_OUTPUT="true"
    SAVE_JSON_OUTPUT="true"
    OUTPUT_JSON_PATH="${OUTPUT_DIR}/result.txt"
    # --------------------------------------

    echo "准备运行检测脚本..."
    echo "  模型权重: ${MODEL_WEIGHTS}"
    echo "  输入源: ${INPUT_SOURCE}"
    echo "  输出目录: ${OUTPUT_DIR}"
    echo "  目标类别: ${TARGET_CLASS}"
    echo ""

    CMD_ARGS=""
    CMD_ARGS+=" --weights ${MODEL_WEIGHTS}"
    CMD_ARGS+=" --source ${INPUT_SOURCE}"

    if [ "${SHOW_OUTPUT}" = "true" ]; then
        CMD_ARGS+=" --show_vid"
    fi

    if [ "${SAVE_OUTPUT}" = "true" ]; then
        CMD_ARGS+=" --save_vid"
        CMD_ARGS+=" --output_dir ${OUTPUT_DIR}"
    fi

    if [ "${SAVE_JSON_OUTPUT}" = "true" ]; then
        CMD_ARGS+=" --save_json"
        CMD_ARGS+=" --output_json_path ${OUTPUT_JSON_PATH}"
    fi

    if [ ! -z "${TARGET_CLASS}" ]; then
        CMD_ARGS+=" --target_class \"${TARGET_CLASS}\""
    fi

    if [ "${SAVE_OUTPUT}" = "true" ] || [ "${SAVE_JSON_OUTPUT}" = "true" ]; then
        mkdir -p "${OUTPUT_DIR}"
        echo "确保输出目录 ${OUTPUT_DIR} 已创建。"
    fi

    python src/process.py ${CMD_ARGS}

    echo ""
    echo "检测脚本执行完毕."
    echo "如果保存了结果，请检查目录: ${OUTPUT_DIR}"

else
    echo ""
    echo "未找到可用的已训练模型 (根目录或 ${TRAIN_OUTPUT_PROJECT_DIR} 中)。将开始训练新模型..."
    echo "训练将使用 project=${TRAIN_OUTPUT_PROJECT_DIR} name=${TRAIN_BASE_NAME}"
    echo "YOLO会自动处理版本（如 ${TRAIN_BASE_NAME}2 ... 如果 ${TRAIN_BASE_NAME} 已存在）"
    echo ""

    # 预训练模型路径 (检查项目根目录)
    BASE_MODEL_YOLO11N_PATH="yolo11n.pt" # Was base_models/yolo11n.pt
    BASE_MODEL_YOLOV8N_PATH="yolov8n.pt" # Was base_models/yolov8n.pt
    TRAIN_MODEL_ARG="yolov8n.pt" # 默认让yolo下载这个

    if [ -f "${BASE_MODEL_YOLO11N_PATH}" ]; then
        echo "在项目根目录发现本地预训练模型: ${BASE_MODEL_YOLO11N_PATH}，将使用此模型开始训练。"
        TRAIN_MODEL_ARG="${BASE_MODEL_YOLO11N_PATH}"
    elif [ -f "${BASE_MODEL_YOLOV8N_PATH}" ]; then
        echo "未找到 ${BASE_MODEL_YOLO11N_PATH}，但在项目根目录发现本地预训练模型: ${BASE_MODEL_YOLOV8N_PATH}，将使用此模型开始训练。"
        TRAIN_MODEL_ARG="${BASE_MODEL_YOLOV8N_PATH}"
    else
        echo "未在项目根目录找到本地预训练模型 ${BASE_MODEL_YOLO11N_PATH} 或 ${BASE_MODEL_YOLOV8N_PATH}。"
        echo "YOLO将尝试自动下载默认基础模型 (${TRAIN_MODEL_ARG})。"
    fi

    yolo train model=${TRAIN_MODEL_ARG} data=dataset.yaml epochs=50 imgsz=640 batch=8 project=${TRAIN_OUTPUT_PROJECT_DIR} name=${TRAIN_BASE_NAME} plots=False

    echo ""
    echo "训练过程已启动/完成。"
    
    LATEST_CREATED_DIR=$(ls -d ${TRAIN_OUTPUT_PROJECT_DIR}/${TRAIN_BASE_NAME}* 2>/dev/null | sort -V | tail -n 1)
    if [ -d "${LATEST_CREATED_DIR}" ]; then
        echo "最新的训练结果保存在: ${LATEST_CREATED_DIR}"
        echo "检查 ${LATEST_CREATED_DIR}/weights/ 内容:"
        ls -l "${LATEST_CREATED_DIR}/weights/"
        echo "下次运行此脚本时，如果模型（best.pt 或 last.pt）存在于此目录或项目根目录，将会自动使用它进行检测。"
    else
        echo "警告: 训练命令执行后，未能定位到新的训练输出目录。"
    fi
fi
# --- 主要逻辑结束 ---

# 注释掉旧的配置和示例，因为它们已被新的逻辑覆盖或不再直接适用
# ... (原有的被注释掉的检测参数和示例命令可以保留或删除，这里省略以保持简洁) ...

echo ""
echo "脚本执行完毕." 