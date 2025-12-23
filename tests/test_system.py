import numpy as np
from unittest import mock

import pytest
from PyQt5.QtWidgets import QApplication

from system import SteelDefectSystem


class DummyBoxes:
    """
    用于模拟 YOLO 返回结果中的 boxes.data 属性。111
    """

    def __init__(self, data: np.ndarray):
        self.data = mock.Mock()
        # 模拟 .cpu().numpy() 调用链
        self.data.cpu.return_value = mock.Mock()
        self.data.cpu.return_value.numpy.return_value = data


class DummyResult:
    """
    用于模拟 YOLO 结果对象，包含 plot() 与 boxes。
    """

    def __init__(self, annotated_frame: np.ndarray, boxes: DummyBoxes):
        self._annotated = annotated_frame
        self.boxes = boxes

    def plot(self):
        return self._annotated


class DummyModel:
    """
    用于替代真实 YOLO 模型的假模型，方便在测试中构造稳定可控的输出。
    """

    def __init__(self):
        # 模拟类别名称映射
        self.names = {0: "defect_a"}

    def __call__(self, frame, conf=0.25, iou=0.45):
        # 构造一个固定的检测结果：[x1, y1, x2, y2, conf, cls_id]
        det = np.array([[10.0, 20.0, 110.0, 220.0, 0.9, 0.0]], dtype=np.float32)
        boxes = DummyBoxes(det)
        annotated = frame.copy()
        return [DummyResult(annotated, boxes)]


@pytest.fixture
def app(qtbot):
    """
    全局 QApplication fixture，pytest-qt 会确保单例应用存在。
    """
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(app, qtbot):
    """
    创建并展示 SteelDefectSystem 主窗口，供 GUI 测试使用。
    """
    window = SteelDefectSystem()
    # 使用假模型替换真实 YOLO，以加速测试并避免依赖权重文件
    window.model = DummyModel()
    qtbot.addWidget(window)
    window.show()
    return window


def test_run_inference_updates_labels_and_table(main_window, qtbot):
    """
    集成测试：验证 run_inference 能够更新界面标签和表格。
    """
    # 构造一张简单的黑色图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated = main_window.run_inference(frame, source_type="Image")
    # annotated 应该与原始尺寸相同
    assert annotated.shape == frame.shape

    # 检查标签是否更新
    assert "目标数目: 1" in main_window.label_count.text()
    assert "类型: defect_a" in main_window.label_current_class.text()
    assert "置信度" in main_window.label_current_conf.text()

    # 检查表格是否添加了记录
    assert main_window.table_widget.rowCount() == 1
    assert main_window.table_widget.item(0, 1).text() == "Image"


def test_display_image_sets_pixmap(main_window, qtbot):
    """
    单元测试：验证 display_image 能够在 QLabel 上显示图像。
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    main_window.display_image(frame)
    assert main_window.image_label.pixmap() is not None


def test_stop_video_stream_releases_capture(main_window):
    """
    单元测试：验证 stop_video_stream 会停止定时器并释放视频流资源。
    """
    # 模拟一个已打开的视频源
    fake_cap = mock.Mock()
    fake_cap.isOpened.return_value = True
    main_window.cap = fake_cap

    # 启动一次定时器以模拟运行状态
    main_window.timer.start(10)
    assert main_window.timer.isActive()

    main_window.stop_video_stream()

    assert not main_window.timer.isActive()
    fake_cap.release.assert_called_once()
    assert main_window.cap is None


def test_close_event_calls_stop_video_stream(main_window, qtbot):
    """
    集成测试：验证窗口关闭事件会调用 stop_video_stream。
    """
    with mock.patch.object(main_window, "stop_video_stream") as mock_stop:
        main_window.close()
        mock_stop.assert_called_once()


