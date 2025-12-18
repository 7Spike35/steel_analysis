import sys
import cv2
import time
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFileDialog, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer

# 导入 YOLO
from ultralytics import YOLO


class SteelDefectSystem(QMainWindow):
    """
    钢材表面缺陷检测系统主窗口。

    该类负责：
    - 初始化并加载 YOLO 模型；
    - 构建 PyQt5 图形界面布局；
    - 处理图片 / 视频 / 摄像头输入；
    - 调用 YOLO 完成检测并在界面上展示结果；
    - 记录检测结果到表格中并支持保存当前检测图像。
    """

    def __init__(self):
        """
        初始化主窗口、加载模型并构建界面。

        注意：
        - 模型权重路径默认使用本地训练得到的 `best.pt`；
        - 如需更换模型或路径，可修改 `self.model` 初始化部分。
        """
        super().__init__()

        # ===========================
        # 1. 初始化配置
        # ===========================
        self.setWindowTitle("基于深度学习的钢材表面缺陷检测系统")
        self.setMinimumSize(1200, 800)

        # 【重要】这里加载模型。
        # 如果要检测钢材，请将此处替换为你训练好的权重文件路径，例如 'runs/detect/train/weights/best.pt'
        # 现在为了演示，我们使用官方的 yolov8n.pt (它会自动下载)
        print("正在加载模型...")
        self.model = YOLO('D:\pytorchusetrue\steel_analysis\\best.pt')
        print("模型加载完成!")

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_image = None  # 保存当前处理的图片

        # ===========================
        # 2. 界面布局设计
        # ===========================
        self.init_ui()

    def init_ui(self):
        """
        初始化并搭建整个主界面布局。

        包含：
        - 左侧图像显示区域；
        - 右侧控制区域（文件导入 / 摄像头 / 结果统计 / 目标位置 / 操作按钮）；
        - 底部检测结果表格。
        """
        # 主体部件
        main_widget = QWidget()
        main_layout = QGridLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # --- 顶部标题 ---
        title_label = QLabel("基于深度学习的钢材表面缺陷检测系统")
        title_label.setFont(QFont("SimHei", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label, 0, 0, 1, 2)

        # --- 左侧：图像显示区 ---
        image_group = QGroupBox("图像显示区域")
        image_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("暂无图像输入，请选择文件或开启摄像头")
        self.image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f0f0f0;")
        self.image_label.setMinimumSize(640, 480)
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group, 1, 0)

        # --- 右侧：控制与结果区 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # 1. 文件导入控制区
        control_group = QGroupBox("文件导入 / 设备控制")
        control_layout = QVBoxLayout()

        btn_image = QPushButton(" 导入图片 (Image)")
        btn_image.setFixedHeight(40)
        btn_image.clicked.connect(self.open_image)

        btn_video = QPushButton(" 导入视频 (Video)")
        btn_video.setFixedHeight(40)
        btn_video.clicked.connect(self.open_video)

        self.btn_camera = QPushButton(" 摄像头开启 (Camera)")
        self.btn_camera.setFixedHeight(40)
        self.btn_camera.setCheckable(True)
        self.btn_camera.clicked.connect(self.toggle_camera)

        control_layout.addWidget(btn_image)
        control_layout.addWidget(btn_video)
        control_layout.addWidget(self.btn_camera)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        # 2. 检测结果统计区
        result_group = QGroupBox("检测结果统计")
        result_layout = QGridLayout()

        self.label_time = QLabel("用时: 0.000 s")
        self.label_count = QLabel("目标数目: 0")
        self.label_current_class = QLabel("类型: 无")
        self.label_current_conf = QLabel("置信度: 0.00%")

        result_layout.addWidget(self.label_time, 0, 0)
        result_layout.addWidget(self.label_count, 0, 1)
        result_layout.addWidget(self.label_current_class, 1, 0)
        result_layout.addWidget(self.label_current_conf, 1, 1)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # 3. 详细坐标区 (简化显示)
        coord_group = QGroupBox("目标位置 (首个目标)")
        coord_layout = QGridLayout()
        self.label_xmin = QLabel("xmin: 0")
        self.label_ymin = QLabel("ymin: 0")
        self.label_xmax = QLabel("xmax: 0")
        self.label_ymax = QLabel("ymax: 0")
        coord_layout.addWidget(self.label_xmin, 0, 0)
        coord_layout.addWidget(self.label_ymin, 0, 1)
        coord_layout.addWidget(self.label_xmax, 1, 0)
        coord_layout.addWidget(self.label_ymax, 1, 1)
        coord_group.setLayout(coord_layout)
        right_layout.addWidget(coord_group)

        # 4. 操作按钮
        op_group = QGroupBox("操作")
        op_layout = QHBoxLayout()
        btn_save = QPushButton("保存结果图")
        btn_save.clicked.connect(self.save_current_image)
        btn_exit = QPushButton("退出系统")
        btn_exit.clicked.connect(self.close)
        op_layout.addWidget(btn_save)
        op_layout.addWidget(btn_exit)
        op_group.setLayout(op_layout)
        right_layout.addWidget(op_group)

        # 添加右侧面板到主布局
        main_layout.addWidget(right_panel, 1, 1)

        # --- 底部：结果表格 ---
        table_group = QGroupBox("检测结果与位置信息记录")
        table_layout = QVBoxLayout()
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(5)
        self.table_widget.setHorizontalHeaderLabels(["序号", "来源类型", "类别", "置信度", "坐标位置 (xyxy)"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_num = 0  # 记录序号

        table_layout.addWidget(self.table_widget)
        table_group.setLayout(table_layout)
        # 占用横向两个格子的宽度
        main_layout.addWidget(table_group, 2, 0, 1, 2)

        # 设置行列比例
        main_layout.setColumnStretch(0, 2)  # 左侧占2份
        main_layout.setColumnStretch(1, 1)  # 右侧占1份
        main_layout.setRowStretch(1, 3)  # 中间图像区占3份高度
        main_layout.setRowStretch(2, 1)  # 底部表格占1份高度

    # ===========================
    # 3. 逻辑处理核心功能
    # ===========================
    def run_inference(self, frame, source_type="Image"):
        """
        执行 YOLO 推理并更新界面上的检测结果相关控件。

        参数：
        - frame: `numpy.ndarray` 类型的 BGR 图像（OpenCV 格式）；
        - source_type: 字符串，标记图像来源（例如 "Image" / "Video/Camera"）。

        返回：
        - annotated_frame: 已绘制检测框和标签的 BGR 图像。
        """
        start_time = time.time()

        # YOLOv8 推理
        # conf=0.25 设置置信度阈值
        results = self.model(frame, conf=0.25, iou=0.45)

        end_time = time.time()
        process_time = end_time - start_time

        # 获取画好框的图像 (BGR格式，用于OpenCV)
        annotated_frame = results[0].plot()
        self.current_image = annotated_frame  # 保存当前结果供保存按钮使用

        # --- 更新界面统计信息 ---
        detections = results[0].boxes.data.cpu().numpy()  # 获取检测框数据 [x1, y1, x2, y2, conf, cls]
        num_objects = len(detections)

        self.label_time.setText(f"用时: {process_time:.3f} s")
        self.label_count.setText(f"目标数目: {num_objects}")

        if num_objects > 0:
            # 取第一个目标的信息显示在详情区
            first_det = detections[0]
            x1, y1, x2, y2, conf, cls_id = first_det
            class_name = self.model.names[int(cls_id)]

            self.label_current_class.setText(f"类型: {class_name}")
            self.label_current_conf.setText(f"置信度: {conf * 100:.2f}%")
            self.label_xmin.setText(f"xmin: {int(x1)}")
            self.label_ymin.setText(f"ymin: {int(y1)}")
            self.label_xmax.setText(f"xmax: {int(x2)}")
            self.label_ymax.setText(f"ymax: {int(y2)}")

            # --- 更新表格 (把所有检测结果写入表格) ---
            for det in detections:
                boxes_xyxy = det[:4].astype(int).tolist()
                conf_val = det[4]
                cls_name_val = self.model.names[int(det[5])]

                self.table_num += 1
                row_idx = self.table_widget.rowCount()
                self.table_widget.insertRow(row_idx)
                self.table_widget.setItem(row_idx, 0, QTableWidgetItem(str(self.table_num)))
                self.table_widget.setItem(row_idx, 1, QTableWidgetItem(source_type))
                self.table_widget.setItem(row_idx, 2, QTableWidgetItem(cls_name_val))
                self.table_widget.setItem(row_idx, 3, QTableWidgetItem(f"{conf_val:.2%}"))
                self.table_widget.setItem(row_idx, 4, QTableWidgetItem(str(boxes_xyxy)))
                self.table_widget.scrollToBottom()
        else:
            # 重置信息
            self.label_current_class.setText("类型: 无")
            self.label_current_conf.setText("置信度: 0.00%")
            self.label_xmin.setText("xmin: 0")

        return annotated_frame

    def display_image(self, frame):
        """
        将 OpenCV BGR 图像转换为 Qt 图像并显示在主界面左侧的 `image_label` 中。

        参数：
        - frame: `numpy.ndarray` 类型的 BGR 图像。
        """
        # OpenCV 的 BGR 转为 RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 保持比例缩放以适应 Label 大小
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    # ===========================
    # 4. 按钮槽函数 (Slots)
    # ===========================
    def open_image(self):
        """
        槽函数：选择并加载一张图片文件，执行检测并在界面中显示。

        - 会先停止当前正在进行的视频流；
        - 通过文件对话框选择图片路径；
        - 成功读取图片后调用 `run_inference` 和 `display_image`。
        """
        self.stop_video_stream()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                processed_frame = self.run_inference(frame, source_type="Image")
                self.display_image(processed_frame)

    def open_video(self):
        """
        槽函数：选择并加载一个视频文件，并以定时器的方式逐帧推理与显示。

        - 会先停止当前正在进行的视频流；
        - 通过文件对话框选择视频文件；
        - 使用 `QTimer` 周期性调用 `update_frame` 完成推理和刷新。"""
        self.stop_video_stream()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)  # 约 30 FPS
            self.btn_camera.setChecked(False)
            self.btn_camera.setText("摄像头开启 (Camera)")

    def toggle_camera(self, checked):
        """
        槽函数：打开或关闭摄像头视频流。

        参数：
        - checked: bool，来自摄像头按钮的选中状态；

        行为：
        - 当选中时：尝试打开默认摄像头并开始定时器；
        - 当取消选中时：停止视频流并重置按钮文字。
        """
        if checked:
            self.stop_video_stream()
            self.cap = cv2.VideoCapture(0)  # 0 代表默认摄像头
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "无法打开摄像头！")
                self.btn_camera.setChecked(False)
                return
            self.timer.start(30)
            self.btn_camera.setText("摄像头关闭 (Camera)")
        else:
            self.stop_video_stream()
            self.btn_camera.setText("摄像头开启 (Camera)")

    def update_frame(self):
        """
        定时器回调函数：从当前视频源读取一帧，执行推理并在界面中显示。

        - 该函数同时适用于视频文件和摄像头；
        - 当读取失败（如播放结束）时会自动停止视频流。
        """
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 视频流在这里进行推理
                processed_frame = self.run_inference(frame, source_type="Video/Camera")
                self.display_image(processed_frame)
            else:
                self.stop_video_stream()

    def stop_video_stream(self):
        """
        停止当前视频流：
        - 停止 `QTimer`；
        - 释放 `cv2.VideoCapture` 对象并清空引用。
        """
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def save_current_image(self):
        """
        将当前检测结果图像 `current_image` 保存到本地文件。

        - 若当前还未完成任何检测（`current_image` 为 None），会弹出警告提示；
        - 若用户在保存对话框中选择了路径，则使用 OpenCV 写入图像。
        """
        if self.current_image is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "result.jpg", "Image Files (*.jpg *.png)")
            if save_path:
                cv2.imwrite(save_path, self.current_image)
                QMessageBox.information(self, "成功", f"图片已保存至: {save_path}")
        else:
            QMessageBox.warning(self, "警告", "当前没有可保存的检测结果图。")

    def closeEvent(self, event):
        """
        重载窗口关闭事件，确保在退出前正确释放视频流资源。

        参数：
        - event: `QCloseEvent`，关闭事件对象。
        """
        self.stop_video_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局字体，尝试解决中文显示问题（视系统而定）
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    window = SteelDefectSystem()
    window.show()
    sys.exit(app.exec_())