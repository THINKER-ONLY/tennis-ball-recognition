import os
import random
import shutil
import argparse
from tqdm import tqdm

def split_dataset(base_images_dir, base_labels_dir, output_images_dir, output_labels_dir, train_ratio=0.8, random_seed=42):
    """
    将数据集划分为训练集和验证集。

    参数:
        base_images_dir (str): 存放所有源图片的目录 (例如 data/processed_yolo_data/images/)
        base_labels_dir (str): 存放所有源YOLO标签的目录 (例如 data/processed_yolo_data/all_labels/)
        output_images_dir (str): 输出图片的目标根目录 (例如 data/processed_yolo_data/images/)
                                 脚本会在此目录下创建 train/ 和 val/ 子目录。
        output_labels_dir (str): 输出标签的目标根目录 (例如 data/processed_yolo_data/labels/)
                                 脚本会在此目录下创建 train/ 和 val/ 子目录。
        train_ratio (float): 训练集所占的比例 (0.0 到 1.0之间)。
        random_seed (int): 随机种子，用于可复现的划分。
    """
    print(f"开始划分数据集...")
    print(f"源图片目录: {base_images_dir}")
    print(f"源标签目录: {base_labels_dir}")
    print(f"目标图片目录: {output_images_dir}")
    print(f"目标标签目录: {output_labels_dir}")
    print(f"训练集比例: {train_ratio:.2f}")

    random.seed(random_seed)

    # 确保输出目录存在
    train_img_path = os.path.join(output_images_dir, "train")
    val_img_path = os.path.join(output_images_dir, "val")
    train_lbl_path = os.path.join(output_labels_dir, "train")
    val_lbl_path = os.path.join(output_labels_dir, "val")

    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(val_img_path, exist_ok=True)
    os.makedirs(train_lbl_path, exist_ok=True)
    os.makedirs(val_lbl_path, exist_ok=True)

    # 获取所有图片文件名 (不含路径，只含扩展名)
    try:
        all_image_files = [f for f in os.listdir(base_images_dir) 
                             if os.path.isfile(os.path.join(base_images_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    except FileNotFoundError:
        print(f"错误: 源图片目录 {base_images_dir} 未找到或无法访问。")
        return
    except Exception as e:
        print(f"读取源图片目录 {base_images_dir} 时发生错误: {e}")
        return

    if not all_image_files:
        print(f"错误: 在 {base_images_dir} 中没有找到图片文件。请确保图片已复制到此目录。")
        return

    print(f"共找到 {len(all_image_files)} 张图片准备划分。")

    # 打乱文件列表
    random.shuffle(all_image_files)

    # 计算分割点
    split_point = int(len(all_image_files) * train_ratio)
    train_files = all_image_files[:split_point]
    val_files = all_image_files[split_point:]

    print(f"训练集图片数量: {len(train_files)}")
    print(f"验证集图片数量: {len(val_files)}")

    # 移动训练集文件
    print("正在处理训练集...")
    for filename in tqdm(train_files, desc="移动训练集文件"):
        img_src = os.path.join(base_images_dir, filename)
        img_dst = os.path.join(train_img_path, filename)
        
        label_filename = os.path.splitext(filename)[0] + ".txt"
        lbl_src = os.path.join(base_labels_dir, label_filename)
        lbl_dst = os.path.join(train_lbl_path, label_filename)

        if not os.path.exists(lbl_src):
            print(f"警告: 图片 {filename} 对应的标签文件 {label_filename} 在 {base_labels_dir} 中未找到。此图片将不会被包含在训练集中。")
            continue
        try:
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)
        except Exception as e:
            print(f"移动文件 {filename} 或其标签时出错: {e}")

    # 移动验证集文件
    print("正在处理验证集...")
    for filename in tqdm(val_files, desc="移动验证集文件"):
        img_src = os.path.join(base_images_dir, filename)
        img_dst = os.path.join(val_img_path, filename)

        label_filename = os.path.splitext(filename)[0] + ".txt"
        lbl_src = os.path.join(base_labels_dir, label_filename)
        lbl_dst = os.path.join(val_lbl_path, label_filename)

        if not os.path.exists(lbl_src):
            print(f"警告: 图片 {filename} 对应的标签文件 {label_filename} 在 {base_labels_dir} 中未找到。此图片将不会被包含在验证集中。")
            continue
        try:
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)
        except Exception as e:
            print(f"移动文件 {filename} 或其标签时出错: {e}")

    print("数据集划分完成！")
    print(f"训练图片已移至: {train_img_path}")
    print(f"训练标签已移至: {train_lbl_path}")
    print(f"验证图片已移至: {val_img_path}")
    print(f"验证标签已移至: {val_lbl_path}")
    print(f"请检查 {base_images_dir} 和 {base_labels_dir} 是否已空 (或只剩下无法匹配标签的图片)。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将图片和YOLO标签划分为训练集和验证集。")
    parser.add_argument("--base_images_dir", type=str, default="data/processed_yolo_data/images",
                        help="包含所有源图片的目录 (这些图片将被移动)")
    parser.add_argument("--base_labels_dir", type=str, default="data/processed_yolo_data/all_labels",
                        help="包含所有源YOLO标签的目录 (这些标签将被移动)")
    parser.add_argument("--output_root_images", type=str, default="data/processed_yolo_data/images",
                        help="输出图片的根目录，脚本会在此创建 train/ 和 val/ (默认与base_images_dir相同，实现原地创建子目录)")
    parser.add_argument("--output_root_labels", type=str, default="data/processed_yolo_data/labels",
                        help="输出标签的根目录，脚本会在此创建 train/ 和 val/ (默认: data/processed_yolo_data/labels)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集所占比例 (例如0.8表示80%%)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于可复现的划分")

    args = parser.parse_args()

    # 根据脚本位置调整相对路径为绝对路径或更可靠的相对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)

    def resolve_path(path_arg):
        if not os.path.isabs(path_arg):
            return os.path.join(project_root, path_arg)
        return path_arg

    base_images_dir = resolve_path(args.base_images_dir)
    base_labels_dir = resolve_path(args.base_labels_dir)
    output_root_images = resolve_path(args.output_root_images)
    output_root_labels = resolve_path(args.output_root_labels)

    split_dataset(base_images_dir, base_labels_dir, output_root_images, output_root_labels, args.train_ratio, args.seed) 