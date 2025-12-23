import os
import shutil
import glob
import random
import yaml
import kagglehub
import torch
from ultralytics import YOLO
from pathlib import Path

# ================= 核心配置 =================
MODEL_VERSION = 'yolo11s.pt'  # 推荐用 Small 版本
IMG_SIZE = 1024               # 高分辨率
EPOCHS = 300                  # 训练轮数
BATCH_SIZE = 4                # 显存小就改小
PROJECT_NAME = 'steel_defect_project'
RUN_NAME = 'multi_best_run'   # 实验名字
# ===========================================

BASE_DIR = os.getcwd()
LOCAL_DIR = os.path.join(BASE_DIR, "datasets", "neu_det")
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# --- (保持原有的数据处理函数不变) ---
def convert_xml_to_yolo(xml_file, output_txt, class_list):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        with open(output_txt, 'w') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in class_list: continue
                cls_id = class_list.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = ((b[0] + b[1]) / 2.0 / w, (b[2] + b[3]) / 2.0 / h,
                      (b[1] - b[0]) / w, (b[3] - b[2]) / h)
                f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")
    except: pass

def prepare_data():
    import xml.etree.ElementTree as ET # 局部引入防报错
    images_train_dir = os.path.join(LOCAL_DIR, 'train', 'images')
    if os.path.exists(images_train_dir) and len(os.listdir(images_train_dir)) > 100:
        print("✅ 本地数据已准备就绪。")
        return LOCAL_DIR
    
    print(f"⬇️ 正在下载数据集...")
    raw_path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
    if os.path.exists(LOCAL_DIR): shutil.rmtree(LOCAL_DIR)
    
    for split in ['train', 'valid']:
        os.makedirs(os.path.join(LOCAL_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(LOCAL_DIR, split, 'labels'), exist_ok=True)
        
    print("🔄 正在转换格式...")
    image_files = []
    for ext in ['*.jpg', '*.bmp', '*.png']:
        image_files.extend(glob.glob(os.path.join(raw_path, '**', ext), recursive=True))
    image_files = list(set(image_files))
    random.shuffle(image_files)
    split_num = int(len(image_files) * 0.8)
    splits = {'train': image_files[:split_num], 'valid': image_files[split_num:]}

    for split, files in splits.items():
        img_dest = os.path.join(LOCAL_DIR, split, 'images')
        lbl_dest = os.path.join(LOCAL_DIR, split, 'labels')
        for img_path in files:
            p = Path(img_path)
            xml = p.with_suffix('.xml')
            if not xml.exists():
                fallback = glob.glob(os.path.join(raw_path, '**', p.stem + '.xml'), recursive=True)
                if fallback: xml = Path(fallback[0])
                else: continue
            shutil.copy(img_path, os.path.join(img_dest, p.name))
            convert_xml_to_yolo(xml, os.path.join(lbl_dest, p.stem + '.txt'), CLASSES)
    return LOCAL_DIR

def create_yaml(dataset_path):
    yaml_path = os.path.join(BASE_DIR, 'neu_det_multi.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path': os.path.abspath(dataset_path),
            'train': 'train/images', 'val': 'valid/images',
            'nc': len(CLASSES), 'names': {i: n for i, n in enumerate(CLASSES)}
        }, f, sort_keys=False)
    return yaml_path

# ========================================================
# 🔥 核心魔法：自定义保存回调函数 🔥
# ========================================================
def on_train_epoch_end(trainer):
    """
    每一轮跑完后，检查各项指标，如果是历史最高，就单独存一份。
    """
    # 1. 初始化历史最高分记录 (如果还没有的话)
    if not hasattr(trainer, 'custom_best_scores'):
        trainer.custom_best_scores = {
            'metrics/recall(B)': 0.0,    # 记录最高 Recall
            'metrics/precision(B)': 0.0, # 记录最高 Precision
            'metrics/mAP50(B)': 0.0      # 记录最高 mAP50
        }

    # 2. 定义我们想要保存的文件名映射
    # key: 指标名称 (YOLO内部名称), value: 保存的文件名
    targets = {
        'metrics/recall(B)': 'best_recall.pt',
        'metrics/precision(B)': 'best_precision.pt',
        'metrics/mAP50(B)': 'best_map50.pt'
    }

    # 3. 遍历指标进行比对
    current_metrics = trainer.metrics
    save_dir = trainer.args.save_dir # 当前训练结果的保存目录
    
    for metric_key, filename in targets.items():
        current_val = current_metrics.get(metric_key, 0.0)
        best_val = trainer.custom_best_scores[metric_key]
        
        # 如果当前轮次的指标 > 历史最高分
        if current_val > best_val:
            trainer.custom_best_scores[metric_key] = current_val # 更新最高分
            
            # 打印好消息
            print(f"\n🌟 [{metric_key}] 创新高! {best_val:.4f} -> {current_val:.4f} | 已保存: {filename}")
            
            # 保存模型
            # 注意：这里我们只保存 weights，不保存优化器状态以节省空间
            save_path = os.path.join(save_dir, 'weights', filename)
            torch.save(trainer.model.state_dict(), save_path)

# ========================================================

def run_training():
    # 1. 准备环境
    ds_path = prepare_data()
    yml_path = create_yaml(ds_path)
    
    # 2. 加载模型
    model = YOLO(MODEL_VERSION)
    
    # 3. 注册我们的回调函数 (这一步很重要！)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    print(f"🚀 开始全能训练模式...")
    print(f"   最终你将在 weights 文件夹下得到 4 个模型：")
    print(f"   1. best.pt (综合最优)")
    print(f"   2. best_recall.pt (查全率最高 - 适合防漏检)")
    print(f"   3. best_precision.pt (查准率最高 - 适合防误报)")
    print(f"   4. best_map50.pt (检测精度最高)")
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    model.train(
        data=yml_path,
        project=PROJECT_NAME,
        name=RUN_NAME,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=50,
        device=device,
        exist_ok=True
    )
    
    print("\n🎉 训练结束！请去 weights 文件夹查看你的战利品。")

if __name__ == "__main__":
    run_training()