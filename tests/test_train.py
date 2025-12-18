import os
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from train import (
    convert_xml_to_yolo,
    create_yaml,
    prepare_data,
    train_model,
    CLASSES,
)


def _create_dummy_xml(path: Path, width: int = 640, height: int = 480) -> None:
    """
    在给定路径创建一个简单的 VOC XML 标注文件，供测试使用。
    """
    content = f"""<annotation>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>
  <object>
    <name>{CLASSES[0]}</name>
    <bndbox>
      <xmin>10</xmin>
      <ymin>20</ymin>
      <xmax>110</xmax>
      <ymax>220</ymax>
    </bndbox>
  </object>
</annotation>"""
    path.write_text(content, encoding="utf-8")


def test_convert_xml_to_yolo_basic():
    """
    单元测试：验证 XML -> YOLO TXT 的基本转换逻辑是否正确。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "test.xml"
        txt_path = Path(tmpdir) / "test.txt"
        _create_dummy_xml(xml_path)

        convert_xml_to_yolo(str(xml_path), str(txt_path), CLASSES)

        assert txt_path.exists()
        content = txt_path.read_text(encoding="utf-8").strip()
        # 检查类别 id 是否为 0，且有 5 个数（cls x y w h）
        parts = content.split()
        assert parts[0] == "0"
        assert len(parts) == 5


@mock.patch("train.kagglehub.dataset_download")
def test_prepare_data_download_and_structure(mock_download):
    """
    集成偏向的单元测试：验证 prepare_data 能够按照预期创建目录结构并调用转换逻辑。
    外部依赖（kagglehub 下载与真实文件结构）通过临时目录和 mock 模拟。
    """
    with tempfile.TemporaryDirectory() as raw_tmp, tempfile.TemporaryDirectory() as local_tmp:
        # 模拟 kagglehub 下载后的目录
        raw_root = Path(raw_tmp)
        img = raw_root / "sample.bmp"
        img.write_bytes(b"fake_image_data")  # 实际内容对测试不重要
        xml = raw_root / "sample.xml"
        _create_dummy_xml(xml)

        mock_download.return_value = str(raw_root)

        # 替换模块中的 LOCAL_DIR 为临时目录，避免污染真实路径
        with mock.patch("train.LOCAL_DIR", os.path.join(local_tmp, "neu_det")):
            dataset_root = prepare_data()

        # 检查目录结构是否建立
        assert os.path.isdir(os.path.join(dataset_root, "train", "images"))
        assert os.path.isdir(os.path.join(dataset_root, "train", "labels"))
        assert os.path.isdir(os.path.join(dataset_root, "valid", "images"))
        assert os.path.isdir(os.path.join(dataset_root, "valid", "labels"))


def test_create_yaml_content_and_path(tmp_path):
    """
    单元测试：验证 create_yaml 生成的 YAML 内容与路径是否正确。
    """
    dataset_path = str(tmp_path / "neu_det_dataset")
    os.makedirs(dataset_path, exist_ok=True)

    # 将当前工作目录切换到临时目录，避免在项目根目录生成文件
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        yaml_path = create_yaml(dataset_path)
        assert os.path.exists(yaml_path)

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["path"] == dataset_path
        assert data["train"] == "train/images"
        assert data["val"] == "valid/images"
        assert data["nc"] == len(CLASSES)
        # names 应该是 {index: name}
        assert isinstance(data["names"], dict)
        assert data["names"][0] == CLASSES[0]
    finally:
        os.chdir(cwd)


@mock.patch("train.YOLO")
def test_train_model_calls_yolo_train(mock_yolo):
    """
    单元测试：验证 train_model 会正确初始化 YOLO 并调用其 train 方法。
    """
    dummy_yaml = "dummy.yaml"
    model_instance = mock.Mock()
    mock_yolo.return_value = model_instance

    train_model(dummy_yaml)

    mock_yolo.assert_called_once()
    model_instance.train.assert_called_once()
    # 确认 data 参数为传入的 yaml 路径
    kwargs = model_instance.train.call_args.kwargs
    assert kwargs["data"] == dummy_yaml


