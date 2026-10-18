"""多相机导出的慢测试：真的写一遍 LeRobot 数据集再读回来。"""

import json
import sys
from pathlib import Path

import pytest

import export_lerobot as ex
import format_dataset as fd
import pipeline_common as pc


def run_main(module, monkeypatch, *args):
    monkeypatch.setattr(
        sys, "argv", [Path(module.__file__).name, *[str(arg) for arg in args]]
    )
    return module.main()


@pytest.mark.slow
def test_multi_camera_export_roundtrip(tmp_path, tiny_jpegs, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    images = tmp_path / "images"
    images.mkdir()
    frames = [0, 6, 12]
    for camera in ("a", "b"):
        for frame in frames:
            (images / pc.frame_filename(f"ep1_{camera}", frame)).write_bytes(
                tiny_jpegs[0].read_bytes()
            )

    meta = {
        "format": 3,
        "tool": "clean_video.py",
        "target_fps": 5,
        "group_by": "dir",
        "source_root": str(tmp_path / "videos"),
        "cameras": ["a", "b"],
        "camera_names": {},
        "episodes": {
            "ep1": {
                "episode_dir": ".",
                "cameras": {"a": "ep1_a", "b": "ep1_b"},
            }
        },
        "videos": {
            "ep1_a": {"orig_fps": 30.0, "source_stem": "a", "episode": "ep1", "camera": "a"},
            "ep1_b": {"orig_fps": 30.0, "source_stem": "b", "episode": "ep1", "camera": "b"},
        },
    }
    (images / pc.IMAGES_META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    score_rows = []
    for camera in ("a", "b"):
        for frame in frames:
            score_rows.append(
                {
                    "video": f"ep1_{camera}",
                    "source_frame": frame,
                    "blur_score": 200.0 if camera == "a" else 50.0,
                }
            )
    (images / pc.BLUR_SCORES_FILENAME).write_text(
        "".join(json.dumps(row) + "\n" for row in score_rows), encoding="utf-8"
    )

    dataset_dir = tmp_path / "dataset"
    assert run_main(fd, monkeypatch, "--input", images, "--output", dataset_dir) == 0

    out = tmp_path / "lerobot"
    assert run_main(
        ex,
        monkeypatch,
        "--input", dataset_dir,
        "--output", out,
        "--dataset-id", "multi_camera_test",
        "--robot-type", "droid",
    ) == 0

    ds = LeRobotDataset("local/embodied-videos", root=out)
    assert len(ds) == 3
    item = ds[0]
    assert item["observation.images.a"].shape[0] == 3
    assert item["observation.images.b"].shape[0] == 3
    assert item["blur_score"].tolist() == pytest.approx([200.0, 50.0])
    assert int(item["source_frame"]) == 0

    manifest = json.loads((out / pc.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "multi_camera_test"
    assert manifest["robot_type"] == "droid"
    assert manifest["num_episodes"] == 1
    assert manifest["num_samples"] == 3
    assert manifest["num_camera_frames"] == {"a": 3, "b": 3}
    assert [camera["name"] for camera in manifest["cameras"]] == ["a", "b"]
    assert manifest["manifest_status"] == "partial"
    assert manifest["content_hash"].startswith("sha256:")
    assert manifest["dataset_version"] == manifest["content_hash"].split(":", 1)[1][:12]
