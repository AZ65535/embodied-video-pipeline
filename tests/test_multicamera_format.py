"""format_dataset.py 的多相机同步与清单契约。"""

import itertools
import json
import logging
from pathlib import Path

import pytest

import format_dataset as fd
import pipeline_common as pc


@pytest.fixture
def build_multi(tmp_path, tiny_jpegs):
    """造一个 dir 模式的第一步产物（图片 + format 3 _meta.json + 分数）。"""
    counter = itertools.count()

    def _build(
        episodes: dict,
        scores: dict = None,
        fps_by_video: dict = None,
        meta_overrides: dict = None,
    ) -> Path:
        root = tmp_path / f"images{next(counter)}"
        root.mkdir()
        payload_episodes = {}
        payload_videos = {}
        all_scores = {}
        global_cameras = None

        for episode, cameras in episodes.items():
            payload_episodes[episode] = {
                "episode_dir": "." if episode == "_root" else episode,
                "cameras": {},
            }
            for camera, frame_numbers in cameras.items():
                prefix = camera if episode == "_root" else f"{episode}_{camera}"
                payload_episodes[episode]["cameras"][camera] = prefix
                payload_videos[prefix] = {
                    "orig_fps": (fps_by_video or {}).get(prefix, 30.0),
                    "source_stem": camera,
                    "episode": episode,
                    "camera": camera,
                }
                for index, frame_no in enumerate(frame_numbers):
                    source = tiny_jpegs[index % len(tiny_jpegs)]
                    (root / pc.frame_filename(prefix, frame_no)).write_bytes(
                        source.read_bytes()
                    )
                per_video = (scores or {}).get(prefix)
                all_scores[prefix] = {
                    frame: (
                        per_video[frame]
                        if per_video is not None and frame in per_video
                        else 250.0
                    )
                    for frame in frame_numbers
                }
            names = sorted(cameras)
            global_cameras = names if global_cameras is None else global_cameras
            assert global_cameras == names

        meta = {
            "format": 3,
            "tool": "clean_video.py",
            "target_fps": 5,
            "group_by": "dir",
            "source_root": str(root.parent / "videos"),
            "cameras": global_cameras,
            "camera_names": {},
            "episodes": payload_episodes,
            "videos": payload_videos,
        }
        meta.update(meta_overrides or {})
        (root / pc.IMAGES_META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        rows = [
            {"video": prefix, "source_frame": frame, "blur_score": score}
            for prefix, frame_scores in all_scores.items()
            for frame, score in frame_scores.items()
        ]
        (root / pc.BLUR_SCORES_FILENAME).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return root

    return _build


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_multi_camera_episode_writes_shared_grid_and_per_camera_rows(
    build_multi, run_format, tmp_path
):
    images = build_multi(
        {"ep1": {"a": [0, 6, 12], "b": [0, 6, 12]}},
        scores={"ep1_a": {0: 300.0, 6: 40.0, 12: 250.0}, "ep1_b": {0: 10.0, 6: 20.0, 12: 30.0}},
    )
    out = tmp_path / "ds"
    assert run_format("--input", images, "--output", out) == 0

    info = read_json(out / "meta" / "info.json")
    assert info["total_frames"] == 3
    assert info["extras"]["total_camera_rows"] == 6
    assert info["extras"]["cameras"] == ["a", "b"]
    assert info["extras"]["group_by"] == "dir"
    assert info["extras"]["source_root"].endswith("videos")
    assert info["features"]["observation.images.a"]["shape"] == info["features"][
        "observation.images.b"
    ]["shape"]
    assert info["features"][pc.BLUR_SCORE_KEY]["shape"] == [2]
    assert info["features"][pc.BLUR_SCORE_KEY]["names"] == ["a", "b"]

    episodes = read_rows(out / "meta" / "episodes.jsonl")
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["length"] == 3
    assert episode["episode"] == "ep1"
    assert episode["cameras"] == ["a", "b"]
    assert episode["source_videos"] == {"a": "ep1_a", "b": "ep1_b"}
    assert episode["source_fps_by_camera"] == {"a": 30.0, "b": 30.0}

    frames = read_rows(out / "meta" / "frames.jsonl")
    assert len(frames) == 6
    by_camera = {}
    for row in frames:
        by_camera.setdefault(row["camera"], []).append(row)
    assert set(by_camera) == {"a", "b"}
    positions_a = [
        (row["episode_index"], row["frame_index"], row["source_frame"], row["timestamp"])
        for row in by_camera["a"]
    ]
    positions_b = [
        (row["episode_index"], row["frame_index"], row["source_frame"], row["timestamp"])
        for row in by_camera["b"]
    ]
    assert positions_a == positions_b
    assert [row["blur_score"] for row in by_camera["a"]] == [300.0, 40.0, 250.0]
    assert [row["is_blurry"] for row in by_camera["a"]] == [False, True, False]
    assert [row["blur_score"] for row in by_camera["b"]] == [10.0, 20.0, 30.0]
    assert len({row["index"] for row in frames}) == 6
    # 中间产物只记录相对源图片目录的文件名
    assert all(Path(row["path"]).parent == Path(".") for row in frames)


def test_multi_camera_frame_count_mismatch_fails(build_multi, run_format, tmp_path, caplog):
    images = build_multi({"ep1": {"a": [0, 6, 12], "b": [0, 6]}})
    with caplog.at_level(logging.ERROR):
        assert run_format("--input", images, "--output", tmp_path / "ds") == 1
    assert "有 2 帧" in caplog.text and "有 3 帧" in caplog.text


def test_multi_camera_source_frame_mismatch_fails(build_multi, run_format, tmp_path, caplog):
    images = build_multi({"ep1": {"a": [0, 6, 12], "b": [0, 3, 12]}})
    with caplog.at_level(logging.ERROR):
        assert run_format("--input", images, "--output", tmp_path / "ds") == 1
    assert "source_frame 序列" in caplog.text
    assert "第 1 个位置" in caplog.text


def test_multi_camera_fps_mismatch_fails(build_multi, run_format, tmp_path, caplog):
    images = build_multi(
        {"ep1": {"a": [0, 6, 12], "b": [0, 6, 12]}},
        fps_by_video={"ep1_b": 25.0},
    )
    with caplog.at_level(logging.ERROR):
        assert run_format("--input", images, "--output", tmp_path / "ds") == 1
    assert "原始帧率不一致" in caplog.text


def test_multi_camera_splits_are_synchronized(build_multi, run_format, tmp_path):
    images = build_multi(
        {"ep1": {"a": list(range(0, 60, 6)), "b": list(range(0, 60, 6))}}
    )
    out = tmp_path / "ds"
    assert run_format(
        "--input", images, "--output", out, "--max-episode-seconds", "1"
    ) == 0

    episodes = read_rows(out / "meta" / "episodes.jsonl")
    assert len(episodes) == 2
    assert {row["segment_index"] for row in episodes} == {0, 1}
    assert {row["episode"] for row in episodes} == {"ep1"}
    assert all(row["length"] == 5 for row in episodes)
    frames = read_rows(out / "meta" / "frames.jsonl")
    assert len(frames) == 20  # 2 段 × 5 个采样 × 2 相机
    for episode_index in {row["episode_index"] for row in frames}:
        segment = [row for row in frames if row["episode_index"] == episode_index]
        grids = {
            camera: [(row["frame_index"], row["source_frame"]) for row in segment if row["camera"] == camera]
            for camera in ("a", "b")
        }
        assert grids["a"] == grids["b"]


def test_camera_name_mapping_renames_features(build_multi, run_format, tmp_path):
    images = build_multi({"ep1": {"18026681": [0, 6], "22008760": [0, 6]}})
    names = tmp_path / "names.json"
    names.write_text(
        json.dumps({"18026681": "left", "22008760": "right"}), encoding="utf-8"
    )
    out = tmp_path / "ds"
    assert run_format(
        "--input", images, "--output", out, "--camera-names", names
    ) == 0
    info = read_json(out / "meta" / "info.json")
    assert info["extras"]["cameras"] == ["left", "right"]
    assert info["extras"]["camera_names"] == {
        "18026681": "left",
        "22008760": "right",
    }
    assert "observation.images.left" in info["features"]
    assert "observation.images.right" in info["features"]


def test_group_by_mismatch_with_meta_fails(build_multi, run_format, tmp_path, caplog):
    images = build_multi({"ep1": {"a": [0, 6], "b": [0, 6]}})
    with caplog.at_level(logging.ERROR):
        assert run_format(
            "--input", images, "--output", tmp_path / "ds", "--group-by", "file"
        ) == 1
    assert "group_by" in caplog.text


def test_unknown_prefix_in_dir_mode_is_skipped(build_multi, run_format, tmp_path, caplog):
    images = build_multi({"ep1": {"a": [0, 6], "b": [0, 6]}})
    # 多出来一组未登记的图片（旧产物残留）
    (images / pc.frame_filename("stray", 0)).write_bytes(
        (images / pc.frame_filename("ep1_a", 0)).read_bytes()
    )
    with caplog.at_level(logging.WARNING):
        assert run_format("--input", images, "--output", tmp_path / "ds") == 0
    assert "未登记的图片前缀" in caplog.text
    frames = read_rows(tmp_path / "ds" / "meta" / "frames.jsonl")
    assert all(row["camera"] in {"a", "b"} for row in frames)
