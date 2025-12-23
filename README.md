## 基于深度学习的钢材表面缺陷检测系统

本项目基于 **PyQt5** 与 **Ultralytics YOLOv8/v11**，实现了一个可视化的钢材表面缺陷检测系统，并配套提供了一键数据准备与模型训练脚本，以及较完整的单元测试与集成测试。

---

### 功能概述

- **缺陷检测可视化客户端（`system.py`）**
  - 使用 PyQt5 构建图形界面。
  - 支持三种输入方式：
    - 导入单张图片；
    - 导入视频文件；
    - 调用本地摄像头实时检测。
  - 集成 YOLOv8/v11 模型，实时输出：
    - 推理耗时；
    - 检测到的目标数量；
    - 当前首个目标的类别、置信度与坐标信息；
    - 所有目标的检测结果记录在表格中（包含来源类型、类别、置信度、坐标）。
  - 支持将当前检测结果图像保存到本地。

- **数据准备与模型训练（`train.py`）**
  - 使用 `kagglehub` 自动下载 NEU surface defect 数据集。
  - 将 VOC XML 标注转换为所需的 TXT 标签格式。
  - 自动划分训练集 / 验证集，并生成 `neu_det_auto.yaml` 配置文件。
  - 调用 Ultralytics YOLOv8 进行训练，并输出权重：
    - 训练结果默认存储于 `runs/detect/steel_defect_auto_run`；
    - 最佳模型权重为 `runs/detect/steel_defect_auto_run/weights/best.pt`。

- **测试（`tests/`）**
  - `tests/test_train.py`：对数据转换、数据准备、配置生成、训练接口进行单元测试（大量使用 mock，避免真实下载和长时间训练）。
  - `tests/test_system.py`：基于 `pytest-qt` 的 GUI 单元测试和集成测试，使用假模型替代真实 YOLO，以保证测试快速稳定。

---

### 环境与依赖

建议使用 Python 3.8+。

安装依赖（在项目根目录执行）：

```bash
pip install -r requirements.txt
```

核心依赖说明：

- `PyQt5`：图形界面框架；
- `ultralytics`：YOLOv8/v11 模型与训练框架；
- `opencv-python`：图像 / 视频读取与处理；
- `kagglehub`：从 Kaggle 下载数据集；
- `pytest`、`pytest-qt`：单元测试与 GUI 测试；
- `PyYAML`：读写 YOLO 配置文件。

---

### 数据准备与模型训练

1. **准备 Kaggle 凭证**

   `train.py` 依赖 `kagglehub` 直接下载数据集，需要你在本地配置好 Kaggle API 凭证（一般是 `~/.kaggle/kaggle.json`）。具体配置方法可参考 Kaggle 官方文档。

2. **运行训练脚本**

   在项目根目录执行：

   ```bash
   python train.py/train2.py(使用更大的模型与更多的训练轮次)
   ```

   脚本会自动完成：

   - 下载并解压数据集；
   - 生成所需的数据目录结构 `datasets/neu_det`；
   - 生成 `neu_det_auto.yaml`；
   - 开始训练，并在 `runs/detect/steel_defect_auto_run` 下保存结果和权重。

3. **获取训练好的权重**

   训练完成后，最佳权重文件路径示例：

   ```text
   runs/detect/steel_defect_auto_run/weights/best.pt
   ```

   你可以将其拷贝 / 替换到 `system.py` 中使用的权重位置，或直接在 `system.py` 中修改模型加载路径。

---

### 图形化检测系统使用说明

1. **确认模型权重路径**

   在 `system.py` 中，模型默认加载路径类似：

   ```python
   self.model = YOLO('D:\\pytorchusetrue\\steel_analysis\\best.pt')
   ```

   请根据你本机的实际权重位置进行修改，例如：

   ```python
   self.model = YOLO('runs/detect/steel_defect_auto_run/weights/best.pt')
   ```

2. **启动界面**

   在项目根目录执行：

   ```bash
   python system.py
   ```

3. **界面操作**

   - **导入图片**：点击“导入图片 (Image)”按钮，选择一张图片，系统会自动进行检测并在左侧显示结果图。
   - **导入视频**：点击“导入视频 (Video)”按钮，选择视频文件，系统会逐帧检测，并在图像区域实时显示检测结果。
   - **摄像头检测**：点击“摄像头开启 (Camera)”按钮，调用默认摄像头，实现实时检测；再次点击可关闭摄像头。
   - **结果统计与坐标**：右侧会显示最近一次检测的耗时、目标数目、首个目标的类别与置信度，以及其坐标信息；底部表格会记录所有检测出的目标信息。
   - **保存结果图**：点击“保存结果图”按钮，可以将当前显示的检测结果图像保存为本地文件。
   - **退出系统**：点击“退出系统”按钮或直接关闭窗口。

---

### 运行测试

在项目根目录执行：

```bash
pytest -q
```

说明：

- `tests/test_train.py`：会使用临时目录和 mock，避免真正下载数据集和耗时训练；
- `tests/test_system.py`：通过 `pytest-qt` 创建 `QApplication` 与主窗口，使用假模型进行推理，验证界面逻辑是否正确（不会真正加载 YOLO 权重）。

---

### 目录结构示例

```text
steel_analysis/
├─ system.py                 # PyQt5 可视化检测系统
├─ train.py                  # 数据准备与模型训练脚本
├─ neu_det_auto.yaml         # 训练用数据配置（运行 train.py 后生成）
├─ datasets/
│  └─ neu_det/               # 处理后的 NEU-DET 数据集（train.py 自动生成）
├─ runs/
│  └─ detect/
│     └─ steel_defect_auto_run/
│        └─ weights/
│           └─ best.pt       # 训练得到的最佳模型权重
├─ tests/
│  ├─ test_train.py          # 训练与数据处理相关测试
│  └─ test_system.py         # GUI 与推理流程相关测试
├─ requirements.txt
└─ README.md
```

---

### 注意事项

- 若更换数据集或类别，请同步修改 `train.py` 中的 `CLASSES` 列表以及相关配置。
- 若在 Windows 上运行摄像头功能，可能需要根据本机摄像头索引修改 `cv2.VideoCapture(0)` 中的参数。
- 若使用 GPU 训练，请确保已正确安装 CUDA 与相应版本的 PyTorch（`ultralytics` 将自动检测与使用）。


