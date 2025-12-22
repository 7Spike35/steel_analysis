import os
import shutil
import random
import glob
import xml.etree.ElementTree as ET
import yaml
import kagglehub
from ultralytics import YOLO
from pathlib import Path

# ================= 配置区域 =================
# Kaggle 数据集标识
DATASET_URL = "kaustubhdikshit/neu-surface-defect-database"
# 处理后的 YOLO 数据集输出目录
LOCAL_DIR = os.path.join(os.getcwd(), "datasets", "neu_det")
# 钢材表面缺陷类别列表，顺序将直接影响 YOLO 标签中的类别 ID
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']


# ===========================================
def convert_xml_to_yolo(xml_file, output_txt, class_list):
    """
    将 VOC 格式的 XML 标注文件转换为 YOLO 所需的 TXT 标签格式。

    参数：
    - xml_file: XML 标注文件路径；
    - output_txt: 输出的 TXT 标签文件路径；
    - class_list: 类别名称列表，索引即类别 ID。

    行为：
    - 从 XML 中读取图像宽高和所有目标框；
    - 将左上角 / 右下角坐标转换为归一化中心坐标 (x, y, w, h)；
    - 按行写入到 TXT 文件中（格式：`cls x y w h`）。
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(output_txt, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_list:
                continue
            cls_id = class_list.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

            # 归一化 xywh
            bb = ((b[0] + b[1]) / 2.0 / w, (b[2] + b[3]) / 2.0 / h,
                  (b[1] - b[0]) / w, (b[3] - b[2]) / h)

            f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


def prepare_data():
    """
    从 Kaggle 下载 NEU-DET 数据集并将其转换为 YOLOv8 可直接使用的目录结构。

    主要步骤：
    1. 使用 `kagglehub.dataset_download` 下载原始数据集；
    2. 按 8:2 划分训练集和验证集；
    3. 为每张图片查找对应的 XML 标注并调用 `convert_xml_to_yolo` 进行转换；
    4. 将图片和标签复制 / 生成到 `LOCAL_DIR` 中的 `train` 与 `valid` 目录。

    返回：
    - 数据集根目录路径 `LOCAL_DIR`。
    """
    print(f"1. 正在从 Kaggle 下载 {DATASET_URL} ...")
    raw_path = kagglehub.dataset_download(DATASET_URL)
    print(f"   下载完成，路径: {raw_path}")
    # 定义目标路径
    images_train_dir = os.path.join(LOCAL_DIR, 'train', 'images')
    labels_train_dir = os.path.join(LOCAL_DIR, 'train', 'labels')
    images_val_dir = os.path.join(LOCAL_DIR, 'valid', 'images')
    labels_val_dir = os.path.join(LOCAL_DIR, 'valid', 'labels')
    # 如果目录已存在，建议清理掉重新生成（防止数据污染），或者手动删除
    if os.path.exists(LOCAL_DIR):
        print("   检测到本地数据集目录已存在，正在清理以确保数据最新...")
        shutil.rmtree(LOCAL_DIR)
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    # 搜索下载下来的文件 (递归查找所有图片)
    # 原始数据集通常结构是 NEU-DET/IMAGES/*.bmp 或 *.jpg
    print("2. 正在搜索并转换数据格式 (XML -> YOLO)...")
    # 支持常见的图片格式
    image_files = []
    for ext in ['*.jpg', '*.bmp', '*.png']:
        image_files.extend(glob.glob(os.path.join(raw_path, '**', ext), recursive=True))
    # 去重
    image_files = list(set(image_files))
    print(f"   找到图片文件: {len(image_files)} 张")
    random.shuffle(image_files)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]


    #转换XML文件并保存
    def process_files(files, img_dest, lbl_dest):
        count = 0
        for img_path in files:
            # 寻找对应的 XML 文件
            # 假设 XML 和图片在同一目录，或者在同级的 ANNOTATIONS 目录
            # 这里的逻辑是：先找同名 XML
            path_obj = Path(img_path)
            xml_path = path_obj.with_suffix('.xml')

            # 如果同级目录下没有 XML，尝试去搜寻整个下载目录下的同名 XML
            if not xml_path.exists():
                # 这种暴力搜索比较慢，但在未知结构的 Kaggle 文件夹中比较稳妥
                possible_xmls = glob.glob(os.path.join(raw_path, '**', path_obj.stem + '.xml'), recursive=True)
                if possible_xmls:
                    xml_path = Path(possible_xmls[0])
                else:
                    # 如果实在找不到标注，就跳过这张图
                    continue

            # 复制图片
            shutil.copy(img_path, os.path.join(img_dest, path_obj.name))

            # 转换并保存 Label
            convert_xml_to_yolo(xml_path, os.path.join(lbl_dest, path_obj.stem + '.txt'), CLASSES)
            count += 1
        return count

    print("   正在处理训练集...")
    t_count = process_files(train_files, images_train_dir, labels_train_dir)
    print("   正在处理验证集...")
    v_count = process_files(val_files, images_val_dir, labels_val_dir)

    print(f"   数据准备完成！训练集: {t_count}, 验证集: {v_count}")
    return LOCAL_DIR


def create_yaml(dataset_path):
    """
    生成 YOLO 训练所需的数据集配置文件 `neu_det_auto.yaml`。

    参数：
    - dataset_path: 已准备好的数据集根目录路径；

    返回：
    - 生成的 yaml 文件路径。
    """
    yaml_content = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'valid/images',
        'nc': len(CLASSES),
        'names': {i: name for i, name in enumerate(CLASSES)}
    }

    yaml_path = os.path.join(os.getcwd(), 'neu_det_auto.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"3. 配置文件已生成: {yaml_path}")
    return yaml_path


def train_model(yaml_path):
    """
    使用 Ultralytics YOLOv8 对钢材表面缺陷数据集进行训练。

    参数：
    - yaml_path: 数据集配置文件路径，一般为 `neu_det_auto.yaml`。

    行为：
    - 加载预训练的 `yolov8n.pt` 模型；
    - 调用 `model.train` 完成训练并将结果保存在 `runs/detect/steel_defect_auto_run` 下。
    """
    print("4. 开始加载 YOLOv8 模型并训练...")
    # 使用 nano 版本快速验证，如果你显卡好，可以改用 'yolov8s.pt'
    model = YOLO('yolov8n.pt')

    model.train(
        data=yaml_path,
        epochs=100,  # 训练轮数
        imgsz=640,  # 图片大小
        batch=16,  # 批次大小
        name='steel_defect_auto_run'  # 结果保存名字
    )
    print("训练全部完成！")
    print(f"最佳模型保存在: runs/detect/steel_defect_auto_run/weights/best.pt")


if __name__ == "__main__":
    # 1. 准备数据
    dataset_path = prepare_data()

    # 2. 生成配置
    yaml_path = create_yaml(dataset_path)

    # 3. 开始训练
    train_model(yaml_path)