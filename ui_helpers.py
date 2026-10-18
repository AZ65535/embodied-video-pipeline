"""Gradio 界面的无 UI 依赖逻辑。

这里不 import gradio / cv2 / lerobot；Pillow 只在生成缩略图时延迟 import。
于是：

- Gradio 没装时，`ui.py` 仍能给出可操作的安装提示；
- 这些逻辑可以在没有 Gradio 的环境里直接单测；
- UI 进程不会因为 import 本模块而拉起重依赖。

命令组装完全复用 `run_pipeline.py`，避免 UI 与 CLI 两套参数解释。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import calibrate_blur as cb
import run_pipeline as rp
from pipeline_common import (
    BLUR_SCORES_FILENAME,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_CAMERA,
    DEFAULT_REPO_ID,
    EPISODES_JSONL,
    FRAMES_JSONL,
    IMAGES_META_FILENAME,
    INFO_JSON,
    META_DIRNAME,
    frame_filename,
    is_blurry,
    natural_key,
    read_jsonl,
)

UI_HOST = "127.0.0.1"
ALL_VIDEOS = "__all__"

THUMB_MAX_WIDTH = 480
THUMB_QUALITY = 75
THUMB_VERSION = 1
THUMB_CACHE_DIR = (
    Path(tempfile.gettempdir()) / "embodied-video-pipeline-ui" / "thumbnails"
)

MAX_LOG_LINES = 1000
MAX_LINE_CHARS = 2000
DISK_SCAN_LIMIT = 200_000
LARGE_SCORES_WARN_BYTES = 200 * 1024 * 1024

STEP_TITLES = {
    1: "抽帧并记录模糊分数",
    2: "生成帧清单",
    3: "导出 LeRobot 数据集",
}


# ---------------------------------------------------------------------------
# 配置与命令规划
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UiConfig:
    input_dir: str = "videos"
    output_base: str = "output"
    dataset_dir: str = "dataset"
    lerobot_dir: str = "lerobot_dataset"
    fps: int = 5
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD
    max_episode_seconds: float = 60.0
    link_mode: str = "auto"
    camera: str = DEFAULT_CAMERA
    task: str = ""
    vcodec: str = "h264"
    repo_id: str = DEFAULT_REPO_ID
    selected_steps: Tuple[int, ...] = (1, 2, 3)
    mode: str = "normal"  # normal | force | skip-existing


@dataclass(frozen=True)
class ResolvedPaths:
    input_dir: Path
    output_base: Path
    images_dir: Path
    dataset_dir: Path
    lerobot_dir: Path


@dataclass
class RunPlan:
    config: UiConfig
    paths: ResolvedPaths
    steps: List[rp.Step]
    selected: List[rp.Step]
    to_run: List[rp.Step]
    skipped: List[rp.Step]
    blocked: List[rp.Step]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.errors and not self.blocked and bool(self.to_run)


def _as_int(value, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def config_from_form(
    input_dir,
    output_base,
    dataset_dir,
    lerobot_dir,
    step1,
    step2,
    step3,
    fps,
    blur_threshold,
    max_episode_seconds,
    link_mode,
    camera,
    task,
    vcodec,
    repo_id,
    mode,
) -> UiConfig:
    """把 Gradio 表单值转成配置；非法数字会变成可被校验捕获的值。"""
    selected = tuple(
        number
        for number, checked in ((1, step1), (2, step2), (3, step3))
        if bool(checked)
    )
    return UiConfig(
        input_dir=str(input_dir or "").strip(),
        output_base=str(output_base or "").strip(),
        dataset_dir=str(dataset_dir or "").strip(),
        lerobot_dir=str(lerobot_dir or "").strip(),
        fps=_as_int(fps, 0),
        blur_threshold=_as_float(blur_threshold, 0.0),
        max_episode_seconds=_as_float(max_episode_seconds, -1.0),
        link_mode=str(link_mode or "").strip(),
        camera=str(camera or "").strip(),
        task=str(task or ""),
        vcodec=str(vcodec or "").strip(),
        repo_id=str(repo_id or "").strip(),
        selected_steps=selected,
        mode=str(mode or "normal"),
    )


def resolve_paths(config: UiConfig) -> ResolvedPaths:
    output_base = rp.resolve_path(config.output_base)
    return ResolvedPaths(
        input_dir=rp.resolve_path(config.input_dir),
        output_base=output_base,
        images_dir=output_base / "images",
        dataset_dir=rp.resolve_path(config.dataset_dir),
        lerobot_dir=rp.resolve_path(config.lerobot_dir),
    )


def images_dir_for(output_base: str) -> Path:
    return rp.resolve_path(output_base) / "images"


def to_namespace(config: UiConfig, paths: ResolvedPaths) -> argparse.Namespace:
    return argparse.Namespace(
        input=str(paths.input_dir),
        output=str(paths.output_base),
        dataset=str(paths.dataset_dir),
        lerobot=str(paths.lerobot_dir),
        fps=config.fps,
        blur_threshold=config.blur_threshold,
        max_episode_seconds=config.max_episode_seconds,
        link_mode=config.link_mode,
        camera=config.camera,
        task=config.task,
        vcodec=config.vcodec,
        repo_id=config.repo_id,
        force=config.mode == "force",
        skip_existing=config.mode == "skip-existing",
        dry_run=False,
    )


def _package_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _validate_plan(
    config: UiConfig,
    paths: ResolvedPaths,
    selected: Sequence[rp.Step],
    to_run: Sequence[rp.Step],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    running = {step.number for step in to_run}

    if not selected:
        errors.append("至少勾选一个步骤。")
    if config.mode not in {"normal", "force", "skip-existing"}:
        errors.append(f"未知的产物处理模式：{config.mode!r}")
    if config.fps <= 0:
        errors.append("fps 必须是正整数。")
    if config.blur_threshold <= 0:
        errors.append("模糊阈值必须大于 0。")
    if config.max_episode_seconds < 0:
        errors.append("episode 时长上限不能为负数（0 表示不切分）。")
    if not config.repo_id:
        errors.append("repo-id 不能为空。")
    if not config.camera:
        errors.append("相机名不能为空。")
    if not config.vcodec:
        errors.append("vcodec 不能为空。")
    if config.link_mode not in {"auto", "hardlink", "symlink", "copy"}:
        errors.append(f"未知的落盘方式：{config.link_mode!r}")

    for label, path in (
        ("输入视频目录", paths.input_dir),
        ("第 1 步输出根目录", paths.output_base),
        ("dataset 目录", paths.dataset_dir),
        ("lerobot 目录", paths.lerobot_dir),
    ):
        if path.exists() and not path.is_dir():
            errors.append(f"{label}已存在但不是目录：{path}")

    if 1 in running and not paths.input_dir.is_dir():
        errors.append(f"输入视频目录不存在：{paths.input_dir}")
    # 前置步骤也在本次运行里时，产物会在执行过程中生成，不能提前要求它存在。
    if 2 in running and 1 not in running and not paths.images_dir.is_dir():
        errors.append(
            f"第 2 步需要图片目录，但它不存在：{paths.images_dir}"
            "（请先跑第 1 步，或勾选第 1 步）"
        )
    if 3 in running and 2 not in running:
        missing = [
            name
            for name in (INFO_JSON, EPISODES_JSONL, FRAMES_JSONL)
            if not (paths.dataset_dir / META_DIRNAME / name).is_file()
        ]
        if missing:
            errors.append(
                f"第 3 步需要 dataset/meta 下的 {', '.join(missing)}；"
                "请先跑第 2 步，或勾选第 2 步"
            )

    if 2 in running and 1 not in running:
        warnings.append("第 2 步将使用现有 images 产物（本次不运行第 1 步）。")
    if 3 in running and 2 not in running:
        warnings.append("第 3 步将使用现有 dataset 产物（本次不运行第 2 步）。")
    if config.mode == "force":
        warnings.append(
            "force 模式：第 3 步会删除并重建整个 lerobot 目录；"
            "第 2 步的 --force 不会清理旧的多余 episode 目录。"
        )
    if config.mode == "skip-existing":
        warnings.append("skip-existing：已完成步骤会被跳过，新增素材不会被处理。")

    if 3 in running:
        info = _read_json(paths.dataset_dir / META_DIRNAME / INFO_JSON)
        if isinstance(info, dict):
            features = info.get("features")
            camera_key = f"observation.images.{config.camera}"
            if isinstance(features, dict) and camera_key not in features:
                warnings.append(
                    f"dataset/info.json 中没有相机特征 {camera_key}；"
                    f"现有特征：{sorted(features)}"
                )

    if 2 in running:
        meta = _read_json(paths.images_dir / IMAGES_META_FILENAME)
        if isinstance(meta, dict):
            recorded = meta.get("target_fps")
            if isinstance(recorded, (int, float)) and recorded != config.fps:
                warnings.append(
                    f"{IMAGES_META_FILENAME} 记录的 target_fps 是 {recorded:g}，"
                    f"与本次 fps {config.fps} 不一致。"
                )

    if 1 in running and not _package_available("cv2"):
        warnings.append("当前环境无法 import cv2；第 1 步的子进程会失败。")
    if 3 in running and not _package_available("lerobot"):
        warnings.append("当前环境无法 import lerobot；第 3 步的子进程会失败。")

    return errors, warnings


def build_plan(config: UiConfig) -> RunPlan:
    paths = resolve_paths(config)
    namespace = to_namespace(config, paths)
    steps = rp.build_steps(
        namespace,
        paths.input_dir,
        paths.output_base,
        paths.dataset_dir,
        paths.lerobot_dir,
    )
    selected = [step for step in steps if step.number in config.selected_steps]
    to_run, skipped, blocked = rp.select_steps(
        selected, namespace.skip_existing, namespace.force
    )
    errors, warnings = _validate_plan(config, paths, selected, to_run)
    return RunPlan(
        config=config,
        paths=paths,
        steps=steps,
        selected=selected,
        to_run=to_run,
        skipped=skipped,
        blocked=blocked,
        errors=errors,
        warnings=warnings,
    )


def render_plan_preview(plan: RunPlan) -> str:
    """命令预览：要执行的命令、跳过/挡下的步骤、校验结果。"""
    lines: List[str] = ["=== 将执行 ==="]
    if plan.to_run:
        for step in plan.to_run:
            lines.append(f"[{step.number}/3] {step.title}")
            lines.append(f"    {rp.format_command(step.cmd)}")
    else:
        lines.append("（没有需要运行的步骤）")

    if plan.skipped:
        lines.append("")
        lines.append("=== 跳过（产物已存在）===")
        for step in plan.skipped:
            lines.append(f"[{step.number}/3] {step.title}")

    if plan.blocked:
        lines.append("")
        lines.append("=== 被挡下（产物已存在）===")
        for step in plan.blocked:
            lines.append(f"[{step.number}/3] {step.title}")
        lines.append("  需要选择 force（覆盖/重建）或 skip-existing（跳过已完成）")

    if plan.errors or plan.warnings:
        lines.append("")
        lines.append("=== 校验 ===")
        for message in plan.errors:
            lines.append(f"错误：{message}")
        for message in plan.warnings:
            lines.append(f"警告：{message}")

    lines.append("")
    if plan.errors:
        lines.append("结论：不能运行（先处理错误）")
    elif plan.blocked:
        lines.append("结论：不能运行（产物已存在，选择 force 或 skip-existing）")
    elif not plan.to_run:
        lines.append("结论：没有需要运行的步骤")
    else:
        lines.append("结论：可以运行")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 数据集摘要
# ---------------------------------------------------------------------------


@dataclass
class DirUsage:
    path: Path
    exists: bool
    size_bytes: int = 0
    file_count: int = 0
    truncated: bool = False
    error: Optional[str] = None


@dataclass
class StepArtifacts:
    number: int
    title: str
    done: bool
    markers: List[Tuple[Path, bool]]


@dataclass
class DatasetSummary:
    paths: ResolvedPaths
    info_error: Optional[str] = None
    total_episodes: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[int] = None
    blur_threshold: Optional[float] = None
    camera_key: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None
    link_mode: Optional[str] = None
    episode_counts: Counter = field(default_factory=Counter)
    episode_frames: Counter = field(default_factory=Counter)
    fps_origins: Counter = field(default_factory=Counter)
    frames_rows: Optional[int] = None
    blurry_frames: Optional[int] = None
    missing_score_frames: Optional[int] = None
    malformed_frame_lines: int = 0
    artifacts: List[StepArtifacts] = field(default_factory=list)
    usages: List[DirUsage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def dir_usage(path: Path, max_files: int = DISK_SCAN_LIMIT) -> DirUsage:
    """表观大小/文件数；不做硬链接去重，最多数 max_files 个文件。"""
    if not path.exists():
        return DirUsage(path=path, exists=False)
    resolved = path.resolve()
    if resolved.parent == resolved:
        return DirUsage(
            path=path,
            exists=True,
            error="拒绝扫描文件系统根目录；请指定具体的数据目录",
        )

    size = 0
    count = 0
    truncated = False
    error: Optional[str] = None
    stack = [path]
    try:
        while stack and not truncated:
            current = stack.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            size += entry.stat(follow_symlinks=False).st_size
                            count += 1
                    except OSError:
                        continue
                    if count >= max_files:
                        truncated = True
                        break
    except OSError as exc:
        error = str(exc)
    return DirUsage(
        path=path,
        exists=True,
        size_bytes=size,
        file_count=count,
        truncated=truncated,
        error=error,
    )


def _summarize_frames(path: Path, summary: DatasetSummary) -> None:
    if not path.is_file():
        return
    rows = 0
    blurry = 0
    missing = 0
    malformed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                rows += 1
                if row.get("is_blurry") is True:
                    blurry += 1
                if row.get("blur_score") is None:
                    missing += 1
    except OSError as exc:
        summary.warnings.append(f"读取 {path.name} 失败：{exc}")
        return
    summary.frames_rows = rows
    summary.blurry_frames = blurry
    summary.missing_score_frames = missing
    summary.malformed_frame_lines = malformed


def summarize_paths(paths: ResolvedPaths, steps: Sequence[rp.Step]) -> DatasetSummary:
    summary = DatasetSummary(paths=paths)
    meta_dir = paths.dataset_dir / META_DIRNAME
    info_path = meta_dir / INFO_JSON
    info = _read_json(info_path)
    if info is None:
        summary.info_error = (
            f"尚未生成或无法解析 {info_path}"
        )
    else:
        summary.total_episodes = info.get("total_episodes")
        summary.total_frames = info.get("total_frames")
        summary.fps = info.get("fps")
        extras = info.get("extras") if isinstance(info.get("extras"), dict) else {}
        summary.blur_threshold = extras.get("blur_score_threshold")
        summary.link_mode = extras.get("link_mode")
        features = info.get("features") if isinstance(info.get("features"), dict) else {}
        camera_key = extras.get("camera_key")
        if not isinstance(camera_key, str) or camera_key not in features:
            camera_key = next(
                (
                    key
                    for key in features
                    if isinstance(key, str)
                    and key.startswith("observation.images.")
                ),
                None,
            )
        summary.camera_key = camera_key
        if isinstance(camera_key, str):
            shape = features.get(camera_key, {}).get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                summary.shape = (shape[0], shape[1])

    episodes_path = meta_dir / EPISODES_JSONL
    episode_rows: List[dict] = []
    if episodes_path.is_file():
        try:
            episode_rows = read_jsonl(episodes_path)
        except ValueError as exc:
            summary.warnings.append(f"{EPISODES_JSONL} 读取失败：{exc}")

    if episode_rows:
        for row in episode_rows:
            name = str(row.get("source_video", "?"))
            summary.episode_counts[name] += 1
            length = row.get("length")
            if isinstance(length, int) and not isinstance(length, bool):
                summary.episode_frames[name] += length
            origin = row.get("source_fps_origin") or "unknown"
            summary.fps_origins[str(origin)] += 1

    if info is not None:
        extras = info.get("extras") if isinstance(info.get("extras"), dict) else {}
        sources = extras.get("sources")
        if not episode_rows and isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue
                name = str(source.get("source_video", "?"))
                summary.episode_counts[name] += 1
                frames = source.get("frames")
                if isinstance(frames, int) and not isinstance(frames, bool):
                    summary.episode_frames[name] += frames
        if not episode_rows:
            origins = extras.get("source_fps_origin_by_video")
            if isinstance(origins, dict):
                for origin in origins.values():
                    summary.fps_origins[str(origin or "unknown")] += 1

    _summarize_frames(meta_dir / FRAMES_JSONL, summary)

    if info is not None:
        if (
            summary.total_frames is not None
            and summary.frames_rows is not None
            and summary.total_frames != summary.frames_rows
        ):
            summary.warnings.append(
                f"info.json 记录 {summary.total_frames} 帧，"
                f"但 frames.jsonl 实际有 {summary.frames_rows} 行。"
            )
        if (
            summary.total_episodes is not None
            and episode_rows
            and summary.total_episodes != len(episode_rows)
        ):
            summary.warnings.append(
                f"info.json 记录 {summary.total_episodes} 个 episode，"
                f"但 episodes.jsonl 有 {len(episode_rows)} 行。"
            )
        if not (meta_dir / FRAMES_JSONL).is_file():
            summary.warnings.append("info.json 存在，但 frames.jsonl 缺失。")
    if summary.malformed_frame_lines:
        summary.warnings.append(
            f"frames.jsonl 有 {summary.malformed_frame_lines} 行无法解析。"
        )
    if summary.link_mode == "hardlink":
        summary.warnings.append(
            "hardlink 模式：dataset 与 images 共享同一批 inode，"
            "下面两者的表观大小会重复计算。"
        )

    for step in steps:
        markers = [(path, path.is_file()) for path in step.done_markers]
        summary.artifacts.append(
            StepArtifacts(
                number=step.number,
                title=step.title,
                done=all(ok for _, ok in markers),
                markers=markers,
            )
        )

    summary.usages = [
        dir_usage(paths.images_dir),
        dir_usage(paths.dataset_dir),
        dir_usage(paths.lerobot_dir),
    ]
    for usage in summary.usages:
        if usage.truncated:
            summary.warnings.append(
                f"{usage.path} 的文件数超过 {DISK_SCAN_LIMIT}，占用统计已截断。"
            )
        if usage.error:
            summary.warnings.append(f"{usage.path} 统计失败：{usage.error}")
    return summary


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def render_summary(summary: DatasetSummary) -> str:
    lines = ["### 数据集摘要（只读，预检用）"]
    if summary.info_error:
        lines.append(f"- **info.json**：{summary.info_error}")

    if summary.total_episodes is not None or summary.total_frames is not None:
        lines.append(
            f"- **episodes**：{summary.total_episodes}　"
            f"**frames**：{summary.total_frames}　"
            f"**fps**：{summary.fps}　"
            f"**阈值**：{summary.blur_threshold}"
        )
    if summary.camera_key:
        # info.json 的 shape 是 [height, width, channels]；展示习惯是宽x高
        shape = f"{summary.shape[1]}x{summary.shape[0]}" if summary.shape else "未知"
        lines.append(f"- **相机**：{summary.camera_key}　**尺寸（宽x高）**：{shape}")
    if summary.link_mode:
        lines.append(f"- **落盘方式**：{summary.link_mode}")
    if summary.frames_rows is not None:
        lines.append(
            f"- **帧清单**：{summary.frames_rows} 行，"
            f"模糊 {summary.blurry_frames}，"
            f"缺分数 {summary.missing_score_frames}，"
            f"坏行 {summary.malformed_frame_lines}"
        )

    if summary.episode_counts:
        lines.append("")
        lines.append("**每源视频**")
        lines.append("")
        lines.append("| source_video | episodes | frames |")
        lines.append("|---|---:|---:|")
        for name in sorted(summary.episode_counts, key=natural_key):
            lines.append(
                f"| {_md_cell(name)} | {summary.episode_counts[name]} | "
                f"{summary.episode_frames[name]} |"
            )

    if summary.fps_origins:
        lines.append("")
        lines.append("**原始帧率来源**")
        lines.append("")
        lines.append("| origin | episodes |")
        lines.append("|---|---:|")
        for origin, count in sorted(summary.fps_origins.items()):
            lines.append(f"| {_md_cell(origin)} | {count} |")

    lines.append("")
    lines.append("**产物检查**")
    lines.append("")
    lines.append("| 步骤 | 状态 | 标记文件 |")
    lines.append("|---|---|---|")
    for artifact in summary.artifacts:
        status = "完成" if artifact.done else "未完成"
        markers = "<br>".join(
            f"{'OK' if ok else '缺失'} {_md_cell(path)}"
            for path, ok in artifact.markers
        )
        lines.append(
            f"| {artifact.number}. {_md_cell(artifact.title)} | {status} | {markers} |"
        )

    lines.append("")
    lines.append("**磁盘占用（表观大小）**")
    lines.append("")
    lines.append("| 目录 | 文件数 | 大小 |")
    lines.append("|---|---:|---:|")
    for usage in summary.usages:
        if not usage.exists:
            lines.append(f"| {_md_cell(usage.path)} | - | 尚未生成 |")
        else:
            suffix = "（截断）" if usage.truncated else ""
            lines.append(
                f"| {_md_cell(usage.path)} | {usage.file_count}{suffix} | "
                f"{_human_size(usage.size_bytes)} |"
            )

    if summary.warnings:
        lines.append("")
        lines.append("**警告**")
        for message in summary.warnings:
            lines.append(f"- {message}")
    if summary.errors:
        lines.append("")
        lines.append("**错误**")
        for message in summary.errors:
            lines.append(f"- {message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 日志与子进程
# ---------------------------------------------------------------------------


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class LogRecord:
    kind: str  # line | progress
    text: str


class LogStreamDecoder:
    """把子进程输出切成永久行与实时进度行。

    tqdm 用裸 `\\r` 刷新进度条；如果直接拼进 Textbox，日志会变成一行超长文本。
    这里把 `\\r`/`\\r\\n`/`\\n` 都当作终止符，`\\r` 产出的记录由 UI 只保留最新一条。
    """

    def __init__(self, max_line_chars: int = MAX_LINE_CHARS) -> None:
        self._buffer = ""
        self._max_line_chars = max_line_chars

    def _emit(self, kind: str, text: str) -> Optional[LogRecord]:
        text = ANSI_RE.sub("", text).strip("\x00")
        if not text:
            return None
        if len(text) > self._max_line_chars:
            text = text[: self._max_line_chars] + "…"
        return LogRecord(kind=kind, text=text)

    def feed(self, text: str) -> List[LogRecord]:
        self._buffer += text
        records: List[LogRecord] = []
        while self._buffer:
            cr = self._buffer.find("\r")
            lf = self._buffer.find("\n")
            if cr == -1 and lf == -1:
                break
            if cr != -1 and (lf == -1 or cr < lf):
                if cr == len(self._buffer) - 1:
                    break  # 等下一个 chunk 判断是 \r 还是 \r\n
                chunk = self._buffer[:cr]
                if self._buffer[cr + 1] == "\n":
                    self._buffer = self._buffer[cr + 2 :]
                    record = self._emit("line", chunk)
                else:
                    self._buffer = self._buffer[cr + 1 :]
                    record = self._emit("progress", chunk)
            else:
                chunk = self._buffer[:lf]
                self._buffer = self._buffer[lf + 1 :]
                record = self._emit("line", chunk)
            if record is not None:
                records.append(record)

        # 没有终止符但已经过长：先吐出一段，避免 buffer 无限增长
        if len(self._buffer) > self._max_line_chars:
            chunk = self._buffer[: self._max_line_chars]
            self._buffer = self._buffer[self._max_line_chars :]
            record = self._emit("line", chunk)
            if record is not None:
                records.append(record)
        return records

    def flush(self) -> List[LogRecord]:
        record = self._emit("line", self._buffer)
        self._buffer = ""
        return [record] if record is not None else []


class LogBuffer:
    def __init__(self, max_lines: int = MAX_LOG_LINES) -> None:
        self._lines: deque = deque(maxlen=max_lines)
        self._progress = ""

    def add(self, record: LogRecord) -> None:
        if record.kind == "progress":
            self._progress = record.text
        else:
            self._lines.append(record.text)
            self._progress = ""

    def add_ui(self, text: str) -> None:
        self._lines.append(f"[ui] {text}")
        self._progress = ""

    def render(self) -> str:
        lines = list(self._lines)
        if self._progress:
            lines.append(self._progress)
        return "\n".join(lines)


class JobBusyError(RuntimeError):
    pass


class JobStartError(RuntimeError):
    pass


class JobRunner:
    """同一时间只允许一个子进程；负责流式读取输出和杀进程树。"""

    def __init__(
        self,
        project_root: Path = rp.PROJECT_ROOT,
        popen_factory=subprocess.Popen,
    ) -> None:
        self._project_root = Path(project_root)
        self._popen = popen_factory
        self._state_lock = threading.Lock()
        self._proc = None
        self._reader: Optional[threading.Thread] = None
        self._queue: Optional[queue.Queue] = None
        self._stop_requested = False
        self._last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def try_start(self, step: rp.Step) -> bool:
        with self._state_lock:
            if self._proc is not None:
                if self._proc.poll() is None:
                    raise JobBusyError("已有任务在运行，请先等待或停止。")
                self._finalize_locked()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            kwargs = dict(
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
            )
            if os.name != "nt":
                kwargs["start_new_session"] = True
            try:
                proc = self._popen(step.cmd, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 交给界面显示
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._proc = None
                raise JobStartError(self._last_error) from exc

            self._proc = proc
            self._stop_requested = False
            self._last_error = None
            self._queue = queue.Queue()
            self._reader = threading.Thread(
                target=self._read_stream,
                args=(proc, self._queue),
                daemon=True,
            )
            self._reader.start()
            return True

    def _read_stream(self, proc, sink: queue.Queue) -> None:
        decoder = LogStreamDecoder()
        try:
            stdout = proc.stdout
            if stdout is None:
                return
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                for record in decoder.feed(text):
                    sink.put(record)
            for record in decoder.flush():
                sink.put(record)
        except Exception as exc:  # noqa: BLE001 - 读取失败也要让界面知道
            sink.put(LogRecord("line", f"[读取子进程输出失败] {exc}"))
        finally:
            sink.put(None)

    def events(self, timeout: float = 0.5) -> Iterator[LogRecord]:
        sink = self._queue
        if sink is None:
            return
        while True:
            try:
                item = sink.get(timeout=timeout)
            except queue.Empty:
                yield LogRecord("tick", "")
                continue
            if item is None:
                return
            yield item

    def wait(self) -> int:
        proc = self._proc
        if proc is None:
            return -1
        code = proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=5)
        return code

    def stop(self) -> str:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return "没有正在运行的任务"
        self._stop_requested = True
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                try:
                    proc.kill()
                except OSError:
                    pass
        else:
            try:
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    proc.kill()
                except OSError:
                    pass
        return "已请求停止"

    def _finalize_locked(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2)
        self._proc = None
        self._reader = None
        self._queue = None

    def close(self) -> None:
        with self._state_lock:
            if self._proc is not None and self._proc.poll() is None:
                # 调用方通常已经 stop；这里兜底，避免留下孤儿进程
                self.stop()
            self._finalize_locked()


JOB_RUNNER = JobRunner()


@dataclass
class RunEvent:
    kind: str  # step_start | line | progress | tick | step_done | failed | stopped | finished
    step: Optional[rp.Step] = None
    text: str = ""
    returncode: Optional[int] = None
    elapsed: float = 0.0


def iter_run(plan: RunPlan, runner: JobRunner) -> Iterator[RunEvent]:
    """按顺序运行 plan.to_run；任一步失败或被停止就不再启动后续步骤。"""
    started_all = time.monotonic()
    for step in plan.to_run:
        step_started = time.monotonic()
        yield RunEvent(
            "step_start",
            step=step,
            text=rp.format_command(step.cmd),
            elapsed=0.0,
        )
        try:
            runner.try_start(step)
        except (JobBusyError, JobStartError) as exc:
            yield RunEvent("failed", step=step, text=str(exc), elapsed=0.0)
            return

        for record in runner.events():
            if record.kind == "tick":
                yield RunEvent(
                    "tick", step=step, elapsed=time.monotonic() - step_started
                )
            else:
                yield RunEvent(
                    record.kind,
                    step=step,
                    text=record.text,
                    elapsed=time.monotonic() - step_started,
                )

        code = runner.wait()
        stopped = runner.stop_requested
        elapsed = time.monotonic() - step_started
        runner.close()

        yield RunEvent(
            "step_done",
            step=step,
            returncode=code,
            elapsed=elapsed,
        )
        if stopped:
            yield RunEvent(
                "stopped",
                step=step,
                returncode=code,
                text="已停止；已生成的产物保留。",
                elapsed=elapsed,
            )
            return
        if code != 0:
            yield RunEvent(
                "failed",
                step=step,
                returncode=code,
                text=f"第 {step.number} 步退出码 {code}",
                elapsed=elapsed,
            )
            return

    yield RunEvent("finished", elapsed=time.monotonic() - started_all)


# ---------------------------------------------------------------------------
# 模糊复核
# ---------------------------------------------------------------------------


@dataclass
class ReviewData:
    images_dir: Path
    scores_path: Path
    signature: Optional[Tuple[str, int, int]] = None
    results: List[cb.VideoScores] = field(default_factory=list)
    total_frames: int = 0
    max_score: float = 0.0
    size_bytes: int = 0
    error: Optional[str] = None
    warning: Optional[str] = None


_review_cache_lock = threading.Lock()
_review_cache: Optional[ReviewData] = None


def _scores_signature(path: Path) -> Optional[Tuple[str, int, int]]:
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def load_review_data(images_dir: Path, use_cache: bool = True) -> ReviewData:
    global _review_cache

    scores_path = images_dir / BLUR_SCORES_FILENAME
    signature = _scores_signature(scores_path)
    if signature is None:
        return ReviewData(
            images_dir=images_dir,
            scores_path=scores_path,
            error=(
                f"找不到 {scores_path}。请先运行第 1 步，"
                "或把「第 1 步输出根目录」指向正确的目录。"
            ),
        )

    if use_cache:
        with _review_cache_lock:
            cached = _review_cache
        if cached is not None and cached.signature == signature:
            return cached

    data = ReviewData(
        images_dir=images_dir,
        scores_path=scores_path,
        signature=signature,
        size_bytes=signature[2],
    )
    if data.size_bytes > LARGE_SCORES_WARN_BYTES:
        data.warning = (
            f"{BLUR_SCORES_FILENAME} 有 {_human_size(data.size_bytes)}；"
            "首次加载可能较慢、占用较多内存。"
        )
    try:
        data.results = cb.load_scores(images_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        data.error = str(exc)
        return data

    data.total_frames = sum(len(result.scores) for result in data.results)
    data.max_score = max(
        (score for result in data.results for score in result.scores), default=0.0
    )
    if use_cache:
        with _review_cache_lock:
            _review_cache = data
    return data


def review_video_names(data: ReviewData) -> List[str]:
    return sorted((result.name for result in data.results), key=natural_key)


@dataclass(frozen=True)
class ReviewFrame:
    video: str
    source_frame: int
    score: float
    src: Path
    status: str  # below | critical | above


@dataclass
class ReviewSelection:
    threshold: float
    band: float
    video: str
    per_video: List[dict] = field(default_factory=list)
    total_frames: int = 0
    total_below: int = 0
    total_critical: int = 0
    filtered_frames: int = 0
    filtered_below: int = 0
    filtered_critical: int = 0
    matched_total: int = 0
    selected: List[ReviewFrame] = field(default_factory=list)
    error: Optional[str] = None


def select_review_frames(
    data: ReviewData,
    threshold: float,
    band: float,
    video: str,
    sample_size: int,
) -> ReviewSelection:
    selection = ReviewSelection(threshold=float(threshold), band=float(band), video=video)
    if data.error:
        selection.error = data.error
        return selection
    if band <= 1:
        selection.error = "临界带倍数必须大于 1。"
        return selection

    high = float(threshold) * float(band)
    per_video: List[dict] = []
    all_pairs: List[Tuple[str, int, float]] = []
    total_below = total_critical = 0
    for result in sorted(data.results, key=lambda item: natural_key(item.name)):
        pairs = list(zip(result.frame_indices, result.scores))
        below = sum(1 for _, score in pairs if is_blurry(score, threshold))
        critical = sum(
            1 for _, score in pairs if threshold <= score <= high
        )
        total_below += below
        total_critical += critical
        per_video.append(
            {
                "name": result.name,
                "frames": len(pairs),
                "below": below,
                "below_pct": (below / len(pairs) * 100) if pairs else 0.0,
                "critical": critical,
            }
        )
        all_pairs.extend((result.name, index, score) for index, score in pairs)

    selection.per_video = per_video
    selection.total_frames = data.total_frames
    selection.total_below = total_below
    selection.total_critical = total_critical

    filtered = [
        pair
        for pair in all_pairs
        if video == ALL_VIDEOS or pair[0] == video
    ]
    selection.filtered_frames = len(filtered)
    selection.filtered_below = sum(
        1 for _, _, score in filtered if is_blurry(score, threshold)
    )
    selection.filtered_critical = sum(
        1 for _, _, score in filtered if threshold <= score <= high
    )
    selection.matched_total = len(filtered)

    ordered = sorted(
        filtered,
        key=lambda pair: (pair[2], natural_key(pair[0]), pair[1]),
    )
    limit = max(1, min(int(sample_size), 500))
    # 对下标取样再映射回去：cb.evenly_spaced 内部会 sorted()，
    # 直接传 (video, index, score) 元组会按视频名重排，丢掉分数顺序。
    positions = cb.evenly_spaced(list(range(len(ordered))), limit)
    sampled = [ordered[position] for position in positions]
    frames: List[ReviewFrame] = []
    for name, index, score in sampled:
        if is_blurry(score, threshold):
            status = "below"
        elif threshold <= score <= high:
            status = "critical"
        else:
            status = "above"
        frames.append(
            ReviewFrame(
                video=name,
                source_frame=index,
                score=score,
                src=data.images_dir / frame_filename(name, index),
                status=status,
            )
        )
    selection.selected = frames
    return selection


STATUS_LABELS = {
    "below": "低于阈值",
    "critical": "临界",
    "above": "高于阈值",
}


def _pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "-"


def render_review_stats(selection: ReviewSelection) -> str:
    if selection.error:
        return f"**无法读取复核数据**：{selection.error}"
    lines = [
        f"**阈值 {selection.threshold:g}**，临界带 "
        f"[{selection.threshold:g}, {selection.threshold * selection.band:g}]",
        "",
        f"全部 {selection.total_frames} 帧：低于阈值 **{selection.total_below}** "
        f"（{_pct(selection.total_below, selection.total_frames)}），"
        f"临界带 {selection.total_critical} 帧。",
    ]
    if selection.video == ALL_VIDEOS:
        lines.append(
            f"当前筛选：全部；图库从 {selection.matched_total} 帧中按分数均匀取样 "
            f"{len(selection.selected)} 帧。"
        )
    else:
        lines.append(
            f"当前筛选：**{_md_cell(selection.video)}**，"
            f"{selection.filtered_frames} 帧中低于阈值 {selection.filtered_below} "
            f"（{_pct(selection.filtered_below, selection.filtered_frames)}），"
            f"临界带 {selection.filtered_critical} 帧；图库取样 "
            f"{len(selection.selected)} 帧。"
        )

    lines.append("")
    lines.append("| 源视频 | 帧数 | 低于阈值 | 占比 | 临界带 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in selection.per_video:
        lines.append(
            f"| {_md_cell(row['name'])} | {row['frames']} | {row['below']} | "
            f"{row['below_pct']:.1f}% | {row['critical']} |"
        )
    lines.append("")
    lines.append(
        "口径：低于阈值是严格 `<`；临界带是 `[阈值, 阈值×倍数]`（含上界）。"
    )
    return "\n".join(lines)


def thumbnail_path(
    src: Path,
    cache_dir: Path = THUMB_CACHE_DIR,
    max_width: int = THUMB_MAX_WIDTH,
) -> Optional[Path]:
    try:
        stat = src.stat()
    except OSError:
        return None
    material = (
        f"v{THUMB_VERSION}|{src.resolve()}|{stat.st_mtime_ns}|"
        f"{stat.st_size}|{max_width}"
    )
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    dst = cache_dir / key[:2] / f"{key}.jpg"
    if dst.is_file():
        return dst

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image  # 延迟 import：没有 Pillow 时其余逻辑仍可用

        tmp = dst.with_name(
            f"{dst.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with Image.open(src) as image:
                rgb = image.convert("RGB")
                rgb.thumbnail((max_width, max_width))
                rgb.save(tmp, format="JPEG", quality=THUMB_QUALITY)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    except Exception:  # noqa: BLE001 - 单张失败不应让整页崩掉
        return None
    return dst


def build_review_gallery(
    selection: ReviewSelection,
    cache_dir: Path = THUMB_CACHE_DIR,
) -> Tuple[List[Tuple[str, str]], int]:
    items: List[Tuple[str, str]] = []
    skipped = 0
    if selection.error:
        return items, skipped
    for frame in selection.selected:
        thumb = thumbnail_path(frame.src, cache_dir=cache_dir)
        if thumb is None:
            skipped += 1
            continue
        caption = (
            f"{frame.score:.1f} | {frame.video} | "
            f"frame {frame.source_frame:06d} | {STATUS_LABELS[frame.status]}"
        )
        items.append((str(thumb), caption))
    return items, skipped


def config_for_format_rerun(
    fallback: UiConfig,
    threshold: float,
    force: bool,
    dataset_dir: Path,
) -> UiConfig:
    """重跑第 2 步时，非阈值参数优先沿用现有 info.json，避免顺手改错别的设置。"""
    info = _read_json(Path(dataset_dir) / META_DIRNAME / INFO_JSON)
    changes = {
        "blur_threshold": float(threshold),
        "selected_steps": (2,),
        "mode": "force" if force else "normal",
    }
    if isinstance(info, dict):
        extras = info.get("extras") if isinstance(info.get("extras"), dict) else {}
        features = info.get("features") if isinstance(info.get("features"), dict) else {}

        fps = info.get("fps")
        if isinstance(fps, int) and fps > 0:
            changes["fps"] = fps

        camera_key = extras.get("camera_key")
        if not isinstance(camera_key, str):
            camera_key = next(
                (
                    key
                    for key in features
                    if isinstance(key, str)
                    and key.startswith("observation.images.")
                ),
                None,
            )
        if isinstance(camera_key, str) and camera_key.startswith(
            "observation.images."
        ):
            changes["camera"] = camera_key[len("observation.images.") :]

        link_mode = extras.get("link_mode")
        if link_mode in {"auto", "hardlink", "symlink", "copy"}:
            changes["link_mode"] = link_mode

        max_seconds = extras.get("max_episode_seconds")
        if isinstance(max_seconds, (int, float)) and not isinstance(max_seconds, bool):
            changes["max_episode_seconds"] = float(max_seconds)

    return replace(fallback, **changes)
