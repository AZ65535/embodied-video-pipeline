"""pipeline_common.py 的跨步骤契约测试（纯标准库）。"""

import json
from pathlib import Path

import pytest

import pipeline_common as pc


# ---------------------------------------------------------------------------
# 相机名与顺序
# ---------------------------------------------------------------------------


def test_camera_key_is_feature_name():
    assert pc.camera_key("wrist") == "observation.images.wrist"


@pytest.mark.parametrize(
    "name",
    ["front", "cam_1", "18026681", "left.wrist"],
)
def test_validate_camera_name_accepts_common_names(name):
    assert pc.validate_camera_name(name) is None


@pytest.mark.parametrize(
    "name, reason",
    [
        ("", "不能为空"),
        (".", "'.'"),
        ("..", "'.'"),
        ("a/b", "路径分隔符"),
        ("a\\b", "路径分隔符"),
        ("a:b", "非法字符"),
        ("front ", "空格或点结尾"),
        ("front.", "空格或点结尾"),
        ("CON", "保留设备名"),
        ("nul", "保留设备名"),
        ("x" * 65, "最长"),
    ],
)
def test_validate_camera_name_rejects_unsafe_names(name, reason):
    assert reason in (pc.validate_camera_name(name) or "")


def test_canonical_camera_order_is_natural_and_case_insensitive():
    assert pc.canonical_camera_order(["cam10", "CAM2", "cam1"]) == [
        "cam1",
        "CAM2",
        "cam10",
    ]


def test_load_camera_names_reads_mapping(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"18026681": "cam_1", "22008760": "cam_2"}), encoding="utf-8")
    assert pc.load_camera_names(path) == {
        "18026681": "cam_1",
        "22008760": "cam_2",
    }


def test_load_camera_names_rejects_non_object(tmp_path):
    path = tmp_path / "names.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 对象"):
        pc.load_camera_names(path)


def test_load_camera_names_rejects_invalid_camera_name(tmp_path):
    path = tmp_path / "names.json"
    path.write_text(json.dumps({"1": "a/b"}), encoding="utf-8")
    with pytest.raises(ValueError, match="路径分隔符"):
        pc.load_camera_names(path)


# ---------------------------------------------------------------------------
# episode key
# ---------------------------------------------------------------------------


def test_episode_key_for_root_is_fixed_name():
    assert pc.episode_key_for_dir(Path(".")) == "_root"


@pytest.mark.parametrize(
    "relative, expected",
    [
        (Path("ep1"), "ep1"),
        (Path("AUTOLab/ep1"), "AUTOLab_ep1"),
        (Path("a/b/ep1"), "a_b_ep1"),
        (Path("a b/ep 1"), "a b_ep 1"),
    ],
)
def test_episode_key_for_dir_flattens_path(relative, expected):
    assert pc.episode_key_for_dir(relative) == expected


def test_episode_key_for_dir_sanitizes_windows_characters():
    key = pc.episode_key_for_dir(Path('a:b/c*d?"e'))
    assert all(char not in key for char in '<>:"/\\|?*')


def test_episode_key_for_dir_prefixes_windows_reserved_name():
    assert pc.episode_key_for_dir(Path("CON")) == "_CON"


def test_episode_key_for_dir_truncates_long_keys_with_stable_hash():
    relative = Path("x" * 200)
    first = pc.episode_key_for_dir(relative)
    second = pc.episode_key_for_dir(relative)
    assert len(first) <= 120
    assert first == second
    assert first.startswith("x" * 100 + "_")


def test_episode_key_for_dir_distinguishes_long_similar_paths():
    first = pc.episode_key_for_dir(Path("x" * 120 + "a"))
    second = pc.episode_key_for_dir(Path("x" * 120 + "b"))
    assert first != second
