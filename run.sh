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
    echo "创建环境后，请先安装依赖: uv pip install ultralytics opencv-python" # tqdm 移除了，除非确实需要
    echo "如果您使用的是标准的 venv，可以尝试:"
    echo "  python3 -m venv .venv"
    echo "创建环境后，请先激活并安装依赖: source .venv/bin/activate && pip install ultralytics opencv-python"
    exit 1
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "已激活 .venv 虚拟环境."
else
    echo "警告: .venv 虚拟环境未激活。请确保已创建并激活虚拟环境，并安装了必要的依赖。"
    echo "依赖包括: ultralytics, opencv-python"
    exit 1
fi
# -------------------------------------

# --- 全局配置 ---
# 用于训练和模型查找
TRAIN_OUTPUT_PROJECT_DIR="tennis_ball_runs"
TRAIN_BASE_NAME="first_train"

# 用于检测 (这些可以在这里统一配置)
INPUT_SOURCE="imgs" # 您可以按需修改这里的 INPUT_SOURCE (已根据要求统一为 imgs)
TARGET_CLASS="tennis_ball"
SHOW_OUTPUT="true"         # "true" 或 "false"
SAVE_OUTPUT="true"         # "true" 或 "false"
SAVE_JSON_OUTPUT="true"    # "true" 或 "false"
PYTHON_SCRIPT_PATH="src/run_cli.py" # Python脚本的路径 (!!! 已修改为新的桥梁脚本 !!!)
# -------------------------------------


# --- 检测函数 ---
# 参数1: 要用于检测的模型权重路径
run_detection() {
    local model_to_run_detection_with="$1"

    if [ -z "${model_to_run_detection_with}" ]; then
        echo "错误 (run_detection): 未提供模型路径进行检测。"
        return 1
    fi
    if [ ! -f "${model_to_run_detection_with}" ]; then
        echo "错误 (run_detection): 模型文件 '${model_to_run_detection_with}' 未找到。"
        return 1
    fi

    echo ""
    echo ">>> 将使用模型 ${model_to_run_detection_with} 进行目标检测 <<<"
    echo ""

    local MODEL_WEIGHTS="${model_to_run_detection_with}"
    local OUTPUT_DIR_DETECT="results/detection_output_$(basename "${MODEL_WEIGHTS}" .pt)"
    local OUTPUT_JSON_PATH_DETECT="${OUTPUT_DIR_DETECT}/result.txt"

    echo "准备运行检测脚本..."
    echo "  模型权重: ${MODEL_WEIGHTS}"
    echo "  输入源: ${INPUT_SOURCE}"
    echo "  输出目录: ${OUTPUT_DIR_DETECT}"
    echo "  目标类别: ${TARGET_CLASS}"
    echo ""

    # 构建参数列表给python命令
    local py_args=() # Initialize an empty array
    py_args+=(--weights "${MODEL_WEIGHTS}")
    py_args+=(--source "${INPUT_SOURCE}")

    if [ "${SHOW_OUTPUT}" = "true" ]; then
        py_args+=(--show_vid)
    fi

    if [ "${SAVE_OUTPUT}" = "true" ]; then
        py_args+=(--save_vid)
        py_args+=(--output_dir "${OUTPUT_DIR_DETECT}")
    fi

    if [ "${SAVE_JSON_OUTPUT}" = "true" ]; then
        py_args+=(--save_json)
        py_args+=(--output_json_path "${OUTPUT_JSON_PATH_DETECT}")
    fi

    if [ ! -z "${TARGET_CLASS}" ]; then
        py_args+=(--target_class "${TARGET_CLASS}")
    fi

    if [ "${SAVE_OUTPUT}" = "true" ] || [ "${SAVE_JSON_OUTPUT}" = "true" ]; then
        mkdir -p "${OUTPUT_DIR_DETECT}"
        echo "确保输出目录 ${OUTPUT_DIR_DETECT} 已创建。"
    fi

    if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then
        echo "错误: Python 检测脚本 '${PYTHON_SCRIPT_PATH}' 未找到!"
        return 1
    fi

    echo "执行命令: python \"${PYTHON_SCRIPT_PATH}\" \"${py_args[@]}\"" # 打印实际执行的命令，方便调试
    python "${PYTHON_SCRIPT_PATH}" "${py_args[@]}" # 使用数组传递参数，更安全

    local exit_code=$? # 获取Python脚本的退出码
    if [ $exit_code -ne 0 ]; then
        echo "警告: Python 检测脚本执行失败，退出码: $exit_code"
        # return $exit_code # 可以选择在这里也返回错误
    fi


    echo ""
    echo "检测脚本执行完毕."
    echo "如果保存了结果，请检查目录: ${OUTPUT_DIR_DETECT}"
    return 0
}
# --- 检测函数结束 ---


# --- 主要逻辑：智能查找模型，然后训练或检测 ---
MODEL_TO_USE_INITIALLY="" # 用于存储初始找到的模型

echo ""
echo "正在查找最新的训练成果 (初始检查)..."

# 优先：检查项目根目录下的模型
if [ -f "best.pt" ]; then
    MODEL_TO_USE_INITIALLY="best.pt"
    echo "在项目根目录找到最佳模型: ${MODEL_TO_USE_INITIALLY}"
elif [ -f "last.pt" ]; then
    MODEL_TO_USE_INITIALLY="last.pt"
    echo "在项目根目录找到最后的模型: ${MODEL_TO_USE_INITIALLY} (未找到 best.pt)"
fi

# 其次：如果根目录没有，则检查 TRAIN_OUTPUT_PROJECT_DIR/ 目录
if [ -z "${MODEL_TO_USE_INITIALLY}" ]; then
    CANDIDATE_DIRS=$(ls -d "${TRAIN_OUTPUT_PROJECT_DIR}/${TRAIN_BASE_NAME}"* 2>/dev/null | sort -V | tail -n 1)
    if [ -d "${CANDIDATE_DIRS}" ]; then
        LATEST_TRAIN_DIR="${CANDIDATE_DIRS}"
        echo "在 ${TRAIN_OUTPUT_PROJECT_DIR} 中找到最新的训练目录: ${LATEST_TRAIN_DIR}"

        BEST_PT_PATH="${LATEST_TRAIN_DIR}/weights/best.pt"
        LAST_PT_PATH="${LATEST_TRAIN_DIR}/weights/last.pt"

        if [ -f "${BEST_PT_PATH}" ]; then
            MODEL_TO_USE_INITIALLY="${BEST_PT_PATH}"
            echo "找到最佳模型: ${MODEL_TO_USE_INITIALLY}"
        elif [ -f "${LAST_PT_PATH}" ]; then
            MODEL_TO_USE_INITIALLY="${LAST_PT_PATH}"
            echo "未找到 best.pt，将使用最后的模型: ${MODEL_TO_USE_INITIALLY}"
        else
            echo "在最新的训练目录 ${LATEST_TRAIN_DIR} 中未找到 best.pt 或 last.pt。"
        fi
    else
        echo "未找到任何 ${TRAIN_BASE_NAME}* 相关的训练目录。"
    fi
fi

# 根据是否找到模型来决定是直接检测还是先训练
if [ ! -z "${MODEL_TO_USE_INITIALLY}" ]; then
    # 模型已存在，直接运行检测
    run_detection "${MODEL_TO_USE_INITIALLY}"
else
    # 模型不存在，先进行训练
    echo ""
    echo ">>> 未找到可用的已训练模型。将开始训练新模型... <<<"
    echo "训练将使用 project=${TRAIN_OUTPUT_PROJECT_DIR} name=${TRAIN_BASE_NAME}"
    echo "YOLO会自动处理版本（如 ${TRAIN_BASE_NAME}2 ... 如果 ${TRAIN_BASE_NAME} 已存在）"
    echo ""

    # 预训练模型路径 (检查项目根目录)
    BASE_MODEL_YOLO11N_PATH="yolo11n.pt"
    BASE_MODEL_YOLOV8N_PATH="yolov8n.pt"
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

    # 执行训练
    if [ ! -f "dataset.yaml" ]; then
        echo "错误: 训练所需的 dataset.yaml 文件未找到！请创建并配置它。"
        exit 1
    fi
    echo "执行训练命令: yolo train model=\"${TRAIN_MODEL_ARG}\" data=dataset.yaml epochs=50 imgsz=640 batch=8 project=\"${TRAIN_OUTPUT_PROJECT_DIR}\" name=\"${TRAIN_BASE_NAME}\" plots=False"
    yolo train model="${TRAIN_MODEL_ARG}" data=dataset.yaml epochs=50 imgsz=640 batch=8 project="${TRAIN_OUTPUT_PROJECT_DIR}" name="${TRAIN_BASE_NAME}" plots=False # 根据需要调整训练参数

    local train_exit_code=$?
    if [ $train_exit_code -ne 0 ]; then
        echo "错误: YOLO 训练过程失败，退出码: $train_exit_code"
        exit $train_exit_code
    fi

    echo ""
    echo "训练过程已完成。"
    echo ""
    echo ">>> 训练完成后，尝试查找新生成的模型并进行检测... <<<"

    # 查找训练后新生成的模型
    MODEL_AFTER_TRAINING=""
    LATEST_CREATED_DIR_AFTER_TRAIN=$(ls -d "${TRAIN_OUTPUT_PROJECT_DIR}/${TRAIN_BASE_NAME}"* 2>/dev/null | sort -V | tail -n 1)

    if [ -d "${LATEST_CREATED_DIR_AFTER_TRAIN}" ]; then
        echo "在 ${TRAIN_OUTPUT_PROJECT_DIR} 中找到最新的训练目录 (训练后): ${LATEST_CREATED_DIR_AFTER_TRAIN}"
        BEST_PT_AFTER_TRAIN="${LATEST_CREATED_DIR_AFTER_TRAIN}/weights/best.pt"
        LAST_PT_AFTER_TRAIN="${LATEST_CREATED_DIR_AFTER_TRAIN}/weights/last.pt"

        if [ -f "${BEST_PT_AFTER_TRAIN}" ]; then
            MODEL_AFTER_TRAINING="${BEST_PT_AFTER_TRAIN}"
            echo "找到新训练的最佳模型: ${MODEL_AFTER_TRAINING}"
        elif [ -f "${LAST_PT_AFTER_TRAIN}" ]; then
            MODEL_AFTER_TRAINING="${LAST_PT_AFTER_TRAIN}"
            echo "未找到新训练的 best.pt，将使用最后的模型 (last.pt): ${MODEL_AFTER_TRAINING}"
        else
            echo "警告: 在最新的训练目录 ${LATEST_CREATED_DIR_AFTER_TRAIN} 中未找到 best.pt 或 last.pt (训练后)。"
        fi
    else
        echo "警告: 训练命令执行后，未能定位到新的训练输出目录 (${TRAIN_OUTPUT_PROJECT_DIR}/${TRAIN_BASE_NAME}*)"
    fi

    # 如果找到了新训练的模型，则用它进行检测
    if [ ! -z "${MODEL_AFTER_TRAINING}" ]; then
        run_detection "${MODEL_AFTER_TRAINING}"
    else
        echo "未能找到新训练的模型进行自动检测。请手动检查训练输出，并使用找到的模型路径重新运行脚本进行检测。"
    fi
fi
# --- 主要逻辑结束 ---

echo ""
echo "脚本执行完毕."