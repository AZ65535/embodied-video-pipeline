"""import_actions.py：动作侧车的 file / dir 两种发现方式。"""

import json
import logging
import sys

import pytest

import import_actions as ia


def touch_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real video")


def touch_h5(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real h5")


def write_episodes(dataset_dir, rows, group_by=None):
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if group_by is not None:
        (meta / "info.json").write_text(
            json.dumps({"extras": {"group_by": group_by}}), encoding="utf-8"
        )


def make_h5(path, rows=5, dim=8):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    with h5py.File(path, "w") as handle:
        handle.create_dataset("action/joint_position", data=np.zeros((rows, dim - 1)))
        handle.create_dataset("action/gripper_position", data=np.zeros((rows, 1)))
        handle.create_dataset(
            "action/robot_state/joint_positions", data=np.ones((rows, dim - 1))
        )
        handle.create_dataset(
            "action/robot_state/gripper_position", data=np.ones((rows, 1))
        )


# ---------------------------------------------------------------------------
# 发现
# ---------------------------------------------------------------------------


def test_find_droid_sources_file_mode_keys_by_video_prefix(tmp_path):
    touch_video(tmp_path / "ep1" / "a.mp4")
    touch_video(tmp_path / "ep1" / "b.mp4")
    touch_h5(tmp_path / "ep1" / "trajectory.h5")
    touch_video(tmp_path / "ep2" / "c.mp4")  # 没有 h5

    sources = ia.find_droid_sources(tmp_path, "file")
    assert set(sources) == {"ep1_a", "ep1_b"}
    assert all(path.name == "trajectory.h5" for path in sources.values())


def test_find_droid_sources_dir_mode_keys_by_episode(tmp_path):
    touch_video(tmp_path / "ep1" / "a.mp4")
    touch_video(tmp_path / "ep1" / "b.mp4")
    touch_h5(tmp_path / "ep1" / "trajectory.h5")
    touch_video(tmp_path / "ep2" / "a.mp4")
    touch_h5(tmp_path / "ep2" / "trajectory.h5")

    sources = ia.find_droid_sources(tmp_path, "dir")
    assert set(sources) == {"ep1", "ep2"}


def test_find_droid_sources_dir_mode_root_episode_is_underscore_root(tmp_path):
    touch_video(tmp_path / "a.mp4")
    touch_video(tmp_path / "b.mp4")
    touch_h5(tmp_path / "trajectory.h5")
    sources = ia.find_droid_sources(tmp_path, "dir")
    assert set(sources) == {"_root"}


def test_find_droid_sources_dir_mode_ignores_dirs_without_mp4(tmp_path):
    touch_h5(tmp_path / "no_video" / "trajectory.h5")
    assert ia.find_droid_sources(tmp_path, "dir") == {}


def test_find_droid_sources_dir_mode_rejects_episode_key_collision(tmp_path):
    # a_b/ 与 a/b/ 都会扁平化成 a_b
    touch_video(tmp_path / "a_b" / "cam.mp4")
    touch_h5(tmp_path / "a_b" / "trajectory.h5")
    touch_video(tmp_path / "a" / "b" / "cam.mp4")
    touch_h5(tmp_path / "a" / "b" / "trajectory.h5")
    with pytest.raises(ValueError, match="同一个 episode 键"):
        ia.find_droid_sources(tmp_path, "dir")


# ---------------------------------------------------------------------------
# CLI 与转换
# ---------------------------------------------------------------------------


def run_main(monkeypatch, *args):
    monkeypatch.setattr(
        sys, "argv", ["import_actions.py", *[str(arg) for arg in args]]
    )
    return ia.main()


def test_group_by_mismatch_fails_before_conversion(tmp_path, monkeypatch, caplog):
    (tmp_path / "videos").mkdir()
    ds = tmp_path / "dataset"
    write_episodes(ds, [{"episode": "ep1"}], group_by="dir")
    with caplog.at_level(logging.ERROR):
        code = run_main(
            monkeypatch,
            "--input", tmp_path / "videos",
            "--dataset", ds,
            "--group-by", "file",
        )
    assert code == 1
    assert "group-by dir" in caplog.text
    assert not (ds / "meta" / "actions").exists()


def test_dir_mode_writes_one_npz_per_episode(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    touch_video(videos / "ep1" / "a.mp4")
    touch_video(videos / "ep1" / "b.mp4")
    make_h5(videos / "ep1" / "trajectory.h5", rows=5)

    ds = tmp_path / "dataset"
    write_episodes(ds, [{"episode": "ep1", "source_video": "a"}], group_by="dir")

    assert run_main(
        monkeypatch,
        "--input", videos,
        "--dataset", ds,
        "--group-by", "dir",
    ) == 0

    files = sorted((ds / "meta" / "actions").glob("*.npz"))
    assert [path.name for path in files] == ["ep1.npz"]
    import numpy as np

    with np.load(files[0]) as data:
        assert data["action"].shape == (5, 8)
        assert data["state"].shape == (5, 8)


def test_run_main_uses_episode_key_not_source_video(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    touch_video(videos / "ep1" / "a.mp4")
    make_h5(videos / "ep1" / "trajectory.h5")

    ds = tmp_path / "dataset"
    write_episodes(
        ds,
        [{"episode": "ep1", "source_video": "ep1_a"}],
        group_by="dir",
    )
    assert run_main(
        monkeypatch,
        "--input", videos,
        "--dataset", ds,
        "--group-by", "dir",
    ) == 0
    assert (ds / "meta" / "actions" / "ep1.npz").is_file()
    assert not (ds / "meta" / "actions" / "ep1_a.npz").exists()


def test_legacy_manifest_without_episode_falls_back_to_source_video(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    touch_video(videos / "a.mp4")
    make_h5(videos / "trajectory.h5")

    ds = tmp_path / "dataset"
    write_episodes(ds, [{"source_video": "a"}])
    assert run_main(
        monkeypatch,
        "--input", videos,
        "--dataset", ds,
        "--group-by", "file",
    ) == 0
    assert (ds / "meta" / "actions" / "a.npz").is_file()
