"""build_dataset_index.py：扫描、冲突规则与旧数据集临时登记。"""

import json
from pathlib import Path

import pytest

import build_dataset_index as bi
import pipeline_common as pc


def write_manifest(
    dataset_dir: Path,
    dataset_id: str,
    dataset_version: str,
    cameras=("a", "b"),
    content_hash: str = "sha256:abc",
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": pc.DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "content_hash": content_hash,
        "robot_type": "droid",
        "cameras": [{"name": name, "shape": [48, 64, 3]} for name in cameras],
        "fps": 5,
        "num_episodes": 2,
        "num_samples": 10,
        "action_dim": 8,
        "manifest_status": "partial",
    }
    path = dataset_dir / pc.MANIFEST_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_legacy_info(dataset_dir: Path, repo_id=None, cameras=("b", "a")) -> Path:
    meta = dataset_dir / "meta"
    meta.mkdir(parents=True)
    data = dataset_dir / "data"
    data.mkdir()
    (data / "file-000.parquet").write_bytes(b"parquet")
    features = {
        f"observation.images.{name}": {"dtype": "video", "shape": [48, 64, 3]}
        for name in cameras
    }
    features["action"] = {"dtype": "float32", "shape": [7]}
    info = {
        "repo_id": repo_id,
        "robot_type": "old",
        "fps": 10,
        "total_episodes": 1,
        "total_frames": 5,
        "features": features,
    }
    path = meta / "info.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    return path


def test_scan_builds_index_entry_from_manifest(tmp_path):
    write_manifest(tmp_path / "droid_autolab", "droid_autolab", "v1")
    index = bi.build_index(tmp_path)
    assert index["schema_version"] == pc.INDEX_SCHEMA_VERSION
    assert index["generated_at"].endswith("Z")
    entry = index["datasets"][0]
    assert entry["dataset_id"] == "droid_autolab"
    assert entry["dataset_version"] == "v1"
    assert entry["path"].endswith("/droid_autolab")
    assert entry["manifest_path"].endswith("/dataset_manifest.json")
    assert entry["manifest_hash"].startswith("sha256:")
    assert entry["cameras"] == ["a", "b"]
    assert entry["num_samples"] == 10
    assert entry["manifest_status"] == "partial"


def test_same_dataset_different_versions_are_allowed(tmp_path):
    write_manifest(tmp_path / "v1", "droid", "v1")
    write_manifest(tmp_path / "v2", "droid", "v2")
    index = bi.build_index(tmp_path)
    assert {(item["dataset_id"], item["dataset_version"]) for item in index["datasets"]} == {
        ("droid", "v1"),
        ("droid", "v2"),
    }


def test_duplicate_dataset_version_at_two_paths_fails(tmp_path):
    write_manifest(tmp_path / "copy1", "droid", "v1", content_hash="sha256:1")
    write_manifest(tmp_path / "copy2", "droid", "v1", content_hash="sha256:1")
    with pytest.raises(ValueError, match="重复的数据集版本"):
        bi.build_index(tmp_path)


def test_same_path_duplicate_is_rejected_by_validation():
    entry = {"dataset_id": "x", "dataset_version": "v1", "path": "D:/x"}
    problems = bi.validate_entries([entry, dict(entry)])
    assert any("同一路径" in problem for problem in problems)


def test_bad_manifest_fails_instead_of_skipping(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / pc.MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="读取失败"):
        bi.build_index(tmp_path)


def test_unsupported_manifest_schema_fails(tmp_path):
    path = write_manifest(tmp_path / "x", "x", "v1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        bi.build_index(tmp_path)


def test_manifest_missing_required_field_fails(tmp_path):
    path = write_manifest(tmp_path / "x", "x", "v1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["content_hash"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content_hash"):
        bi.build_index(tmp_path)


def test_infer_legacy_registers_old_dataset_as_partial(tmp_path):
    write_manifest(tmp_path / "new", "new", "v1")
    write_legacy_info(tmp_path / "old", repo_id="old/repo")
    index = bi.build_index(tmp_path, infer_legacy=True)
    by_id = {entry["dataset_id"]: entry for entry in index["datasets"]}
    assert set(by_id) == {"new", "old/repo"}
    legacy = by_id["old/repo"]
    assert legacy["manifest_status"] == "partial"
    assert legacy["manifest_path"] is None
    assert legacy["content_hash"] is None
    assert legacy["dataset_version"].startswith("legacy-")
    assert legacy["cameras"] == ["a", "b"]  # 规范顺序
    assert legacy["action_dim"] == 7
    assert legacy["fps"] == 10


def test_infer_legacy_uses_directory_name_without_repo_id(tmp_path):
    write_legacy_info(tmp_path / "old_dataset", repo_id=None)
    index = bi.build_index(tmp_path, infer_legacy=True)
    assert index["datasets"][0]["dataset_id"] == "old_dataset"


def test_infer_legacy_does_not_duplicate_manifest_datasets(tmp_path):
    write_manifest(tmp_path / "ds", "ds", "v1")
    write_legacy_info(tmp_path / "ds", repo_id="ds")
    index = bi.build_index(tmp_path, infer_legacy=True)
    assert len(index["datasets"]) == 1


def test_infer_legacy_is_opt_in(tmp_path):
    write_legacy_info(tmp_path / "old", repo_id="old")
    assert bi.build_index(tmp_path)["datasets"] == []


def test_infer_legacy_ignores_intermediate_manifest_only_dataset(tmp_path):
    """第二步的 dataset/ 只有 meta/（可能还有空 data/），不能被误登记。"""
    dataset = tmp_path / "intermediate"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (dataset / "data").mkdir()
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0-manifest", "features": {}}),
        encoding="utf-8",
    )
    assert bi.build_index(tmp_path, infer_legacy=True)["datasets"] == []


def test_index_is_rebuildable(tmp_path):
    write_manifest(tmp_path / "ds", "ds", "v1")
    first = bi.build_index(tmp_path)
    second = bi.build_index(tmp_path)
    assert first["datasets"] == second["datasets"]
