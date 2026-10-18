#!/usr/bin/env python
"""扫描多个 LeRobot 数据集，生成可重建的 datasets.json 索引。

    <数据集根目录>/ ──build_dataset_index.py──▶ datasets.json

`dataset_manifest.json` 是每个数据集的**唯一权威自描述**；这个索引只是给训练侧
快速发现数据集的缓存，删掉随时可以重建。索引不扫描视频/图片，只读 manifest
（必要时读旧数据集的 meta/info.json）。

用法：
    python build_dataset_index.py --root D:/datasets --output D:/datasets/datasets.json
    python build_dataset_index.py --root D:/datasets --infer-legacy   # 旧数据集临时登记
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline_common import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    INDEX_FILENAME,
    INDEX_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    META_DIRNAME,
    atomic_write_json,
    canonical_camera_order,
    setup_logging,
)

logger = logging.getLogger("build_dataset_index")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def read_manifest(path: Path) -> dict:
    """读并做最小校验；字段缺失时明确报错，不猜。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"读不了 manifest {path}：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的内容不是 JSON 对象")
    version = payload.get("schema_version")
    if version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} 的 schema_version={version!r}，本工具只支持 "
            f"{DATASET_MANIFEST_SCHEMA_VERSION}"
        )
    for field in ("dataset_id", "dataset_version", "content_hash"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path} 缺少有效字段 {field}")
    return payload


def entry_from_manifest(path: Path, manifest: dict) -> dict:
    data_root = path.parent
    cameras = manifest.get("cameras") or []
    if not isinstance(cameras, list):
        cameras = []
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "content_hash": manifest["content_hash"],
        "path": data_root.resolve().as_posix(),
        "manifest_path": path.resolve().as_posix(),
        "manifest_hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "robot_type": manifest.get("robot_type"),
        "cameras": [
            camera.get("name")
            for camera in cameras
            if isinstance(camera, dict) and isinstance(camera.get("name"), str)
        ],
        "fps": manifest.get("fps"),
        "num_episodes": manifest.get("num_episodes"),
        "num_samples": manifest.get("num_samples"),
        "action_dim": manifest.get("action_dim"),
        "manifest_status": manifest.get("manifest_status", "partial"),
    }


def _feature_cameras(info: dict) -> List[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    names = [
        key[len("observation.images.") :]
        for key in features
        if isinstance(key, str) and key.startswith("observation.images.")
    ]
    return canonical_camera_order(names)


def _feature_action_dim(info: dict) -> Optional[int]:
    features = info.get("features")
    if not isinstance(features, dict):
        return None
    action = features.get("action")
    shape = action.get("shape") if isinstance(action, dict) else None
    if isinstance(shape, list) and shape and isinstance(shape[0], int):
        return shape[0]
    return None


def entry_from_legacy(dataset_dir: Path, info_path: Path) -> dict:
    """从旧数据集的 meta/info.json 生成 partial 条目。

    仅用于临时登记；dataset_id 优先用 repo_id，缺失时才退到目录名并告警。
    版本号用 info.json 内容哈希，避免所有旧数据集都叫同一个 'unknown'。
    """
    try:
        info_bytes = info_path.read_bytes()
        info = json.loads(info_bytes.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"读不了旧数据集 {info_path}：{exc}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"{info_path} 的内容不是 JSON 对象")

    repo_id = info.get("repo_id")
    if isinstance(repo_id, str) and repo_id.strip():
        dataset_id = repo_id
    else:
        dataset_id = dataset_dir.name
        logger.warning(
            "旧数据集 %s 的 info.json 没有 repo_id，临时用目录名 %r 作 dataset_id",
            dataset_dir,
            dataset_id,
        )
    return {
        "dataset_id": dataset_id,
        "dataset_version": "legacy-" + hashlib.sha256(info_bytes).hexdigest()[:12],
        "content_hash": None,
        "path": dataset_dir.resolve().as_posix(),
        "manifest_path": None,
        "manifest_hash": None,
        "robot_type": info.get("robot_type"),
        "cameras": _feature_cameras(info),
        "fps": info.get("fps"),
        "num_episodes": info.get("total_episodes"),
        "num_samples": info.get("total_frames"),
        "action_dim": _feature_action_dim(info),
        "manifest_status": "partial",
    }


def collect_legacy(root: Path, known_dirs: set) -> List[dict]:
    entries = []
    for info_path in sorted(root.rglob(f"{META_DIRNAME}/info.json")):
        dataset_dir = info_path.parent.parent
        if dataset_dir.resolve() in known_dirs:
            continue
        # 中间产物 dataset/ 也可能有空的 data/ 和 meta/info.json；
        # 真正的 LeRobot 数据集一定有 parquet。没有就跳过，避免误登记。
        data_dir = dataset_dir / "data"
        if not data_dir.is_dir() or next(data_dir.rglob("*.parquet"), None) is None:
            continue
        logger.warning("旧数据集没有 dataset_manifest.json，按 partial 登记：%s", dataset_dir)
        entries.append(entry_from_legacy(dataset_dir, info_path))
    return entries


def validate_entries(entries: List[dict]) -> List[str]:
    """冲突规则：同 (id, version) 只能在一个路径；同 id 不同版本允许并存。"""
    problems: List[str] = []
    by_version: Dict[Tuple[str, str], dict] = {}
    by_path: Dict[str, dict] = {}
    for entry in entries:
        key = (entry["dataset_id"], entry["dataset_version"])
        existing = by_version.get(key)
        if existing is not None:
            problems.append(
                f"重复的数据集版本 {key[0]}@{key[1]}："
                f"{existing['path']} 与 {entry['path']}"
            )
        else:
            by_version[key] = entry

        path = entry["path"]
        existing_path = by_path.get(path)
        if existing_path is not None:
            problems.append(
                f"同一路径出现多个数据集记录：{path}"
                f"（{existing_path['dataset_id']}@{existing_path['dataset_version']} 与 "
                f"{entry['dataset_id']}@{entry['dataset_version']}）"
            )
        else:
            by_path[path] = entry
    return problems


def build_index(root: Path, infer_legacy: bool = False) -> dict:
    if not root.is_dir():
        raise ValueError(f"数据集根目录不存在：{root}")

    manifest_paths = sorted(root.rglob(MANIFEST_FILENAME))
    entries: List[dict] = []
    failures: List[str] = []
    known_dirs = set()
    for path in manifest_paths:
        known_dirs.add(path.parent.resolve())
        try:
            entries.append(entry_from_manifest(path, read_manifest(path)))
        except (OSError, ValueError) as exc:
            failures.append(str(exc))

    if failures:
        raise ValueError(
            "以下 manifest 读取失败，未生成索引：\n  " + "\n  ".join(failures)
        )

    if infer_legacy:
        entries.extend(collect_legacy(root, known_dirs))

    problems = validate_entries(entries)
    if problems:
        raise ValueError(
            "数据集索引存在冲突，未生成索引：\n  " + "\n  ".join(problems)
        )

    entries.sort(key=lambda item: (item["dataset_id"], item["dataset_version"]))
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "datasets": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 dataset_manifest.json，生成可重建的 datasets.json 索引。"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="数据集根目录（递归查找 dataset_manifest.json）",
    )
    parser.add_argument(
        "--output",
        default=INDEX_FILENAME,
        help=f"索引输出路径（默认 {INDEX_FILENAME}）",
    )
    parser.add_argument(
        "--infer-legacy",
        action="store_true",
        help="把没有 manifest 的旧数据集按 partial 临时登记（会告警）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    try:
        index = build_index(root, infer_legacy=args.infer_legacy)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_write_json(output, index, logger):
        logger.error("写入索引失败：%s", output)
        return 1

    logger.info(
        "=== 完成 === 登记 %d 个数据集版本（%d 个逻辑数据集）→ %s",
        len(index["datasets"]),
        len({entry["dataset_id"] for entry in index["datasets"]}),
        output.resolve(),
    )
    for entry in index["datasets"]:
        logger.info(
            "  %s@%s  %s  episodes=%s samples=%s",
            entry["dataset_id"],
            entry["dataset_version"],
            entry["path"],
            entry["num_episodes"],
            entry["num_samples"],
        )
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
