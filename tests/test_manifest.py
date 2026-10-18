"""dataset_manifest.json 的字段、计数和 content hash 契约。"""

import argparse
import json
import logging
from pathlib import Path

import pytest

import export_lerobot as ex
import pipeline_common as pc


def make_args(**overrides) -> argparse.Namespace:
    values = dict(
        dataset_id="droid_autolab",
        dataset_version=None,
        repo_id="local/embodied-videos",
        robot_type="droid",
        action_space_id="unknown",
        state_space_id="unknown",
        control_mode="unknown",
        stats_path="meta/stats.json",
        split_path=None,
        split_rules="by_episode",
        recommended_split="train",
        suggested_sampling_weight=1.0,
        fps=5,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def make_episode(episode_index=0, episode="ep1", cameras=("a", "b"), length=3):
    frames_by_camera = {
        camera: [
            {
                "episode_index": episode_index,
                "frame_index": position,
                "source_frame": position * 6,
                "timestamp": position * 0.2,
                "camera": camera,
                "blur_score": 250.0,
                "path": f"{episode}_{camera}_{position}.jpg",
            }
            for position in range(length)
        ]
        for camera in cameras
    }
    return ex.EpisodeFrames(
        episode_index=episode_index,
        episode=episode,
        source_video=cameras[0],
        cameras=list(cameras),
        source_fps=30.0,
        frames_by_camera=frames_by_camera,
        length=length,
    )


@pytest.fixture
def manifest_env(tmp_path):
    out = tmp_path / "lerobot"
    meta = out / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 5,
        "features": {
            "observation.images.a": {"dtype": "video", "shape": [48, 64, 3]},
            "observation.images.b": {"dtype": "video", "shape": [24, 32, 3]},
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (meta / "stats.json").write_text("{}", encoding="utf-8")
    (out / "data").mkdir()
    (out / "data" / "file-000.parquet").write_bytes(b"parquet")
    cameras = [
        ex.CameraSpec("a", "observation.images.a", 48, 64),
        ex.CameraSpec("b", "observation.images.b", 24, 32),
    ]
    intermediate_info = {
        "extras": {
            "source_root": "D:/videos",
            "group_by": "dir",
            "camera_names": {"1": "a", "2": "b"},
            "image_root": "../images",
            "sources": [
                {
                    "cameras": [
                        {"name": "a", "orig_fps": 30.0},
                        {"name": "b", "orig_fps": 30.0},
                    ]
                }
            ],
        }
    }
    return out, cameras, [make_episode()], intermediate_info


def build_manifest(env, args=None):
    out, cameras, episodes, info = env
    return ex.build_dataset_manifest(
        args or make_args(), info, cameras, episodes, 8, 3, out
    )


def test_manifest_records_identity_counts_and_cameras(manifest_env):
    manifest = build_manifest(manifest_env)
    assert manifest["schema_version"] == pc.DATASET_MANIFEST_SCHEMA_VERSION
    assert manifest["dataset_id"] == "droid_autolab"
    assert manifest["repo_id"] == "local/embodied-videos"
    assert manifest["robot_type"] == "droid"
    assert manifest["format_version"] == "v3.0"
    assert manifest["num_episodes"] == 1
    assert manifest["num_samples"] == 3
    assert manifest["num_camera_frames"] == {"a": 3, "b": 3}
    assert [camera["name"] for camera in manifest["cameras"]] == ["a", "b"]
    assert [camera["shape"] for camera in manifest["cameras"]] == [
        [48, 64, 3],
        [24, 32, 3],
    ]
    assert [camera["source_fps"] for camera in manifest["cameras"]] == [30.0, 30.0]
    assert manifest["source"] == {
        "root": "D:/videos",
        "group_by": "dir",
        "camera_names": {"1": "a", "2": "b"},
        "image_root": "../images",
    }
    assert manifest["action_dim"] == 8
    assert manifest["state_dim"] == 8
    assert manifest["stats_status"] == "complete"
    assert manifest["dataset_version"] == manifest["content_hash"].split(":", 1)[1][:12]
    assert manifest["created_at"].endswith("Z")


def test_manifest_is_partial_until_semantics_are_known(manifest_env):
    manifest = build_manifest(manifest_env)
    assert manifest["manifest_status"] == "partial"
    assert manifest["action_space_id"] == "unknown"
    assert manifest["units"] is None
    assert manifest["joint_order"] is None


def test_manifest_honors_explicit_version_and_id(manifest_env):
    manifest = build_manifest(
        manifest_env,
        make_args(dataset_id="my_dataset", dataset_version="v7"),
    )
    assert manifest["dataset_id"] == "my_dataset"
    assert manifest["dataset_version"] == "v7"


def test_manifest_content_hash_is_stable_across_identical_calls(manifest_env):
    first = build_manifest(manifest_env)
    second = build_manifest(manifest_env)
    assert first["content_hash"] == second["content_hash"]
    assert first["dataset_version"] == second["dataset_version"]


def test_manifest_content_hash_changes_when_files_change(manifest_env):
    first = build_manifest(manifest_env)
    out = manifest_env[0]
    (out / "data" / "file-000.parquet").write_bytes(b"different size")
    second = build_manifest(manifest_env)
    assert first["content_hash"] != second["content_hash"]


def test_manifest_content_hash_changes_when_metadata_changes(manifest_env):
    first = build_manifest(manifest_env)
    second = build_manifest(manifest_env, make_args(robot_type="other"))
    assert first["content_hash"] != second["content_hash"]


def test_manifest_reports_missing_stats(manifest_env):
    out = manifest_env[0]
    (out / "meta" / "stats.json").unlink()
    manifest = build_manifest(manifest_env)
    assert manifest["stats_status"] == "missing"


def test_manifest_rejects_missing_split_path(manifest_env):
    with pytest.raises(ValueError, match="split-path"):
        build_manifest(manifest_env, make_args(split_path="nope.json"))


def test_write_dataset_manifest_strips_internal_fields_and_writes(manifest_env, caplog):
    out, cameras, episodes, info = manifest_env
    logger = logging.getLogger("test_manifest")
    with caplog.at_level(logging.WARNING):
        assert ex.write_dataset_manifest(
            make_args(),
            info,
            cameras,
            episodes,
            8,
            3,
            out,
            logger,
        ) == 0
    path = out / pc.MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "droid_autolab"
    assert "_unknown_fields" not in manifest
    assert "_stats_missing" not in manifest
    assert "partial" in caplog.text
