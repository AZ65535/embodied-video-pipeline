#!/usr/bin/env python
"""把 clean_video.py 抽出的帧整理成清单（中间产物）。

管线第二步，产物 `dataset/meta/` 供 export_lerobot.py 消费。帧图片仍留在第一步的
`output/images/`，本脚本只写清单和路径，不复制/链接图片。仅依赖 Python 标准库
（未引入 pyarrow / lerobot / 任何图像库）。
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pipeline_common import (
    BLUR_SCORES_FILENAME,
    DEFAULT_CAMERA,
    EPISODES_JSONL,
    FPS_ORIGIN_INFERRED,
    FPS_ORIGIN_NOMINAL,
    FPS_ORIGIN_RECORDED,
    FRAMES_JSONL,
    IMAGES_META_FILENAME,
    INFO_JSON,
    META_DIRNAME,
    METADATA_PREFIX,
    TASKS_JSONL,
    EpisodeRow,
    FrameRow,
    camera_key,
    canonical_camera_order,
    has_eoi,
    is_blurry,
    load_camera_names,
    natural_key,
    parse_frame_name,
    probe_jpeg,
    read_blur_scores,
    setup_logging,
    validate_camera_name,
    write_json,
    write_jsonl,
)

@dataclass
class FrameEntry:
    """输入目录里的一帧原图。宽高在扫描阶段读出，避免落盘前重复读文件。"""

    src: Path
    source_frame: int
    width: int
    height: int
    camera: str = DEFAULT_CAMERA
    blur_score: Optional[float] = None
    is_blurry: Optional[bool] = None


@dataclass
class EpisodePlan:
    """一个 episode/segment 及其每一路相机的同步帧。"""

    episode_index: int
    episode: str
    episode_dir: str
    cameras: List[str]
    camera_frames: Dict[str, List[FrameEntry]]
    source_prefixes: Dict[str, str]
    source_fps_by_camera: Dict[str, Optional[float]] = field(default_factory=dict)
    source_fps_origin_by_camera: Dict[str, Optional[str]] = field(default_factory=dict)
    source_fps: Optional[float] = None
    source_fps_origin: Optional[str] = None
    observed_step: Optional[int] = None
    segment_index: int = 0
    segment_count: int = 1

    @property
    def frames(self) -> List[FrameEntry]:
        """主相机帧，供报告/兼容代码使用。"""
        return self.camera_frames[self.cameras[0]]

    @property
    def source_video(self) -> str:
        return self.source_prefixes[self.cameras[0]]

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass
class Counters:
    """扫描阶段的异常计数，最后汇总打印。"""

    unparsed: int = 0
    duplicate: int = 0
    corrupt: int = 0
    unassigned: int = 0


def load_images_meta(input_dir: Path) -> dict:
    """读取 clean_video.py 写下的 _meta.json；缺失或损坏时返回空字典。"""
    logger = logging.getLogger("format_dataset")
    meta_path = input_dir / IMAGES_META_FILENAME
    if not meta_path.is_file():
        logger.info("未找到 %s，timestamp 将退回名义值（frame_index / fps）", meta_path)
        return {}

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("%s 无法解析，忽略", meta_path)
        return {}
    if not isinstance(payload, dict):
        logger.warning("%s 的内容不是 JSON 对象，忽略", meta_path)
        return {}

    logger.info("已读取 %s（%d 个视频的原始帧率记录）", meta_path, len(payload.get("videos") or {}))
    return payload


def extract_source_fps(images_meta: dict) -> Dict[str, float]:
    """从 _meta.json 中抽出 {视频名: 原始帧率}，忽略非法条目。"""
    result: Dict[str, float] = {}
    videos = images_meta.get("videos")
    if not isinstance(videos, dict):
        return result
    for name, entry in videos.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("orig_fps")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            result[str(name)] = float(value)
    return result


def load_blur_scores(input_dir: Path) -> Dict[str, Dict[int, float]]:
    """读取 clean_video.py 写下的 _scores.jsonl，返回 {视频名: {原始帧号: 分数}}。

    文件缺失时返回空字典而不是报错：改造前的产物目录没有这个文件，模糊信息就只
    能记为未知，不该让整个流程跑不下去。
    """
    logger = logging.getLogger("format_dataset")
    scores_path = input_dir / BLUR_SCORES_FILENAME
    if not scores_path.is_file():
        logger.info(
            "未找到 %s，帧清单里的 blur_score 将为空（上游是旧版 clean_video.py？）",
            scores_path,
        )
        return {}

    result = read_blur_scores(scores_path, logger)
    logger.info(
        "已读取 %s（%d 个视频、%d 帧的模糊分数）",
        scores_path,
        len(result),
        sum(len(v) for v in result.values()),
    )
    return result


def infer_frame_step(frames: List[FrameEntry]) -> Optional[int]:
    """从原始帧号的间隔反推抽帧步长。

    上游按固定步长采样，相邻帧的帧号差必是步长的整数倍。帧号若有缺口（稀疏目录、
    手工裁剪的素材），缺口只会把差值抬成倍数，所以对全部差值取 gcd 就能还原步长。
    """
    if len(frames) < 2:
        return None
    step = 0
    for prev, cur in zip(frames, frames[1:]):
        step = math.gcd(step, cur.source_frame - prev.source_frame)
        if step == 1:
            break  # gcd 不可能再变小
    return step or None


def resolve_camera_fps(
    frames: List[FrameEntry],
    source_video: str,
    target_fps: int,
    meta_fps: Dict[str, float],
) -> Tuple[Optional[float], str, Optional[int]]:
    """决定某一路相机的原始帧率，返回 (fps, 来源, 观测步长)。"""
    logger = logging.getLogger("format_dataset")
    observed_step = infer_frame_step(frames)

    recorded = meta_fps.get(source_video)
    if recorded:
        expected_step = max(1, int(round(recorded / float(target_fps))))
        if observed_step and observed_step != expected_step:
            logger.warning(
                "%s：%s 记录的帧率 %g 对应步长 %d，但从帧号反推的步长是 %d，两者不符"
                "（素材与 %s 可能对不上），仍按记录值处理",
                source_video,
                IMAGES_META_FILENAME,
                recorded,
                expected_step,
                observed_step,
                IMAGES_META_FILENAME,
            )
        return recorded, FPS_ORIGIN_RECORDED, observed_step

    if observed_step:
        inferred = observed_step * float(target_fps)
        logger.warning(
            "%s：%s 中没有该视频的帧率记录，按帧号间隔反推为 %g fps（步长 %d × %d）。"
            "这是近似值（误差约 ±%g fps）；要精确值请重跑 clean_video.py 生成 %s",
            source_video,
            IMAGES_META_FILENAME,
            inferred,
            observed_step,
            target_fps,
            target_fps / 2.0,
            IMAGES_META_FILENAME,
        )
        return inferred, FPS_ORIGIN_INFERRED, observed_step

    logger.warning(
        "%s：既无 %s 记录、也无法从帧号反推帧率（只有 %d 帧），"
        "将使用名义时间戳 frame_index/%d —— 该时间轴按保留帧数计，帧号有缺口时会偏小",
        source_video,
        IMAGES_META_FILENAME,
        len(frames),
        target_fps,
    )
    return None, FPS_ORIGIN_NOMINAL, None


def resolve_episode_fps(
    episode: EpisodePlan, args: argparse.Namespace, meta_fps: Dict[str, float]
) -> Tuple[Optional[float], str]:
    """兼容包装：解析主相机帧率。"""
    fps, origin, step = resolve_camera_fps(
        episode.frames, episode.source_video, args.fps, meta_fps
    )
    episode.observed_step = step
    return fps, origin


def split_segments(frames: List[FrameEntry], max_frames: int) -> List[List[FrameEntry]]:
    """把一段帧按固定长度切成多段；max_frames <= 0 表示不切。

    切分不是为了绕开某个技术限制（视频存储下没有单 episode 的大小上限），
    而是为了训练时的样本独立性：episode 越长，一个 batch 里来自同一段连续
    视频的样本越多，时间上越相关。生态里常见的 episode 是几秒到几十秒。
    """
    if max_frames <= 0 or len(frames) <= max_frames:
        return [frames]
    return [frames[i : i + max_frames] for i in range(0, len(frames), max_frames)]


def scan_input(
    input_dir: Path,
    counters: Counters,
    target_fps: int,
    blur_scores: Optional[Dict[str, Dict[int, float]]] = None,
    blur_threshold: float = 0.0,
    max_frames_per_episode: int = 0,
    group_by: Optional[str] = None,
    default_camera: str = DEFAULT_CAMERA,
    images_meta: Optional[dict] = None,
    camera_names: Optional[Dict[str, str]] = None,
    meta_fps: Optional[Dict[str, float]] = None,
) -> List[EpisodePlan]:
    """扫描图片并按 episode/相机分组、校验同步、切分。"""
    logger = logging.getLogger("format_dataset")
    scores = blur_scores or {}
    camera_names = camera_names or {}
    meta_fps = meta_fps or {}
    meta = images_meta or {}
    meta_format = meta.get("format")
    meta_group_by = meta.get("group_by") if meta_format == 3 else "file"
    if group_by is not None and meta_format == 3 and group_by != meta_group_by:
        raise ValueError(
            f"_meta.json 的 group_by={meta_group_by!r}，"
            f"与本次 --group-by {group_by!r} 不一致"
        )
    actual_group_by = group_by or meta_group_by or "file"
    if actual_group_by == "dir" and meta_format != 3:
        raise ValueError(
            "--group-by dir 需要 format 3 的 _meta.json；"
            "请用 clean_video.py --group-by dir 重跑第一步"
        )

    prefix_meta: Dict[str, Tuple[str, str, str, str]] = {}
    if meta_format == 3 and isinstance(meta.get("episodes"), dict):
        videos_meta = meta.get("videos") if isinstance(meta.get("videos"), dict) else {}
        for episode, info in meta["episodes"].items():
            if not isinstance(info, dict):
                continue
            cameras_map = info.get("cameras")
            if not isinstance(cameras_map, dict):
                continue
            episode_dir = str(info.get("episode_dir", "."))
            for camera, prefix in cameras_map.items():
                if not isinstance(camera, str) or not isinstance(prefix, str):
                    continue
                video_info = videos_meta.get(prefix)
                stem = (
                    str(video_info.get("source_stem", prefix))
                    if isinstance(video_info, dict)
                    else prefix
                )
                prefix_meta[prefix] = (str(episode), camera, episode_dir, stem)

    grouped: Dict[str, Dict[int, Tuple[Path, int, int]]] = defaultdict(dict)
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(METADATA_PREFIX):
            continue

        parsed = parse_frame_name(path)
        if parsed is None:
            logger.warning("跳过：文件名不符合 {视频名}_frame_{帧号}.jpg 规范 -> %s", path.name)
            counters.unparsed += 1
            continue

        video_name, source_frame = parsed
        if source_frame in grouped[video_name]:
            logger.warning(
                "跳过重复帧：%s 的帧号 %d 已由 %s 提供，忽略 %s",
                video_name,
                source_frame,
                grouped[video_name][source_frame][0].name,
                path.name,
            )
            counters.duplicate += 1
            continue

        if path.stat().st_size == 0:
            logger.warning("跳过：空文件 -> %s", path.name)
            counters.corrupt += 1
            continue

        size = probe_jpeg(path)
        if size is None:
            logger.warning("跳过：JPEG 结构损坏或无法解析宽高 -> %s", path.name)
            counters.corrupt += 1
            continue

        if not has_eoi(path):
            logger.warning("注意：文件末尾缺少 EOI 标记，可能被截断 -> %s", path.name)

        grouped[video_name][source_frame] = (path, size[0], size[1])

    per_episode: Dict[str, Dict[str, List[FrameEntry]]] = {}
    episode_dirs: Dict[str, str] = {}
    source_prefixes: Dict[str, Dict[str, str]] = {}

    for prefix in sorted(grouped, key=natural_key):
        if prefix in prefix_meta:
            episode, recorded_camera, episode_dir, stem = prefix_meta[prefix]
        elif actual_group_by == "file":
            episode, recorded_camera, episode_dir, stem = prefix, default_camera, ".", prefix
        else:
            counters.unassigned += len(grouped[prefix])
            logger.warning(
                "dir 模式下发现未登记的图片前缀 %s，跳过 %d 帧",
                prefix,
                len(grouped[prefix]),
            )
            continue

        camera = camera_names.get(stem, recorded_camera)
        reason = validate_camera_name(camera)
        if reason:
            raise ValueError(f"{prefix} 的相机名 {camera!r} 非法：{reason}")
        cameras_map = per_episode.setdefault(episode, {})
        if camera in cameras_map:
            raise ValueError(f"{episode} 的相机名重复：{camera}")
        video_scores = scores.get(prefix, {})
        frames: List[FrameEntry] = []
        for frame_no, (path, width, height) in sorted(grouped[prefix].items()):
            score = video_scores.get(frame_no)
            frames.append(
                FrameEntry(
                    src=path,
                    source_frame=frame_no,
                    width=width,
                    height=height,
                    camera=camera,
                    blur_score=score,
                    is_blurry=None if score is None else is_blurry(score, blur_threshold),
                )
            )
        cameras_map[camera] = frames
        episode_dirs[episode] = episode_dir
        source_prefixes.setdefault(episode, {})[camera] = prefix

    if not per_episode:
        return []

    camera_sets = {frozenset(names) for names in per_episode.values()}
    if len(camera_sets) > 1:
        detail = "; ".join(
            f"{episode}: {sorted(names)}" for episode, names in sorted(per_episode.items())
        )
        raise ValueError(f"不同 episode 的相机集合不一致：{detail}")
    cameras = canonical_camera_order(next(iter(camera_sets)))
    declared = meta.get("cameras") if meta_format == 3 else None
    if isinstance(declared, list) and set(declared) == set(cameras):
        cameras = list(declared)

    plans: List[EpisodePlan] = []
    episode_index = 0
    for episode in sorted(per_episode, key=natural_key):
        cameras_map = per_episode[episode]
        primary = cameras[0]
        primary_frames = cameras_map[primary]
        n = len(primary_frames)
        reference_frames = [frame.source_frame for frame in primary_frames]

        for camera in cameras:
            frames = cameras_map[camera]
            if len(frames) != n:
                raise ValueError(
                    f"{episode} 的相机 {camera} 有 {len(frames)} 帧，"
                    f"主相机 {primary} 有 {n} 帧"
                )
            got = [frame.source_frame for frame in frames]
            if got != reference_frames:
                position = next(
                    index
                    for index, (left, right) in enumerate(zip(reference_frames, got))
                    if left != right
                )
                raise ValueError(
                    f"{episode} 的相机 {camera} 与 {primary} 的 source_frame 序列"
                    f"在第 {position} 个位置不一致：{got[position]} vs {reference_frames[position]}"
                )
            dims = Counter((frame.width, frame.height) for frame in frames)
            if len(dims) > 1:
                logger.warning(
                    "%s/%s 内部有多种图片尺寸（%s）；第三步会拒绝尺寸不符的帧",
                    episode,
                    camera,
                    dict(dims),
                )

        fps_by_camera: Dict[str, Optional[float]] = {}
        origin_by_camera: Dict[str, Optional[str]] = {}
        observed_steps: Dict[str, Optional[int]] = {}
        for camera in cameras:
            fps, origin, step = resolve_camera_fps(
                cameras_map[camera],
                source_prefixes[episode][camera],
                target_fps,
                meta_fps,
            )
            fps_by_camera[camera] = fps
            origin_by_camera[camera] = origin
            observed_steps[camera] = step

        known_rates = {round(value, 6) for value in fps_by_camera.values() if value is not None}
        if len(known_rates) > 1:
            raise ValueError(f"{episode} 各相机原始帧率不一致：{fps_by_camera}")
        episode_fps = next(
            (value for value in fps_by_camera.values() if value is not None), None
        )
        if episode_fps is not None:
            missing = [camera for camera, value in fps_by_camera.items() if value is None]
            if missing:
                logger.warning(
                    "%s 的部分相机没有原始帧率（%s）；使用其他相机的 %g",
                    episode,
                    ", ".join(missing),
                    episode_fps,
                )
        origins = list(origin_by_camera.values())
        if all(origin == FPS_ORIGIN_RECORDED for origin in origins):
            aggregate_origin = FPS_ORIGIN_RECORDED
        elif FPS_ORIGIN_NOMINAL in origins:
            aggregate_origin = FPS_ORIGIN_NOMINAL
        else:
            aggregate_origin = FPS_ORIGIN_INFERRED
        steps = {step for step in observed_steps.values() if step is not None}
        if len(steps) > 1:
            raise ValueError(f"{episode} 各相机观测步长不一致：{observed_steps}")
        observed_step = next(iter(steps), None)

        if max_frames_per_episode <= 0 or n <= max_frames_per_episode:
            ranges = [(0, n)]
        else:
            ranges = [
                (start, min(start + max_frames_per_episode, n))
                for start in range(0, n, max_frames_per_episode)
            ]

        for segment_index, (start, end) in enumerate(ranges):
            plans.append(
                EpisodePlan(
                    episode_index=episode_index,
                    episode=episode,
                    episode_dir=episode_dirs[episode],
                    cameras=list(cameras),
                    camera_frames={
                        camera: cameras_map[camera][start:end] for camera in cameras
                    },
                    source_prefixes={
                        camera: source_prefixes[episode][camera] for camera in cameras
                    },
                    source_fps_by_camera=dict(fps_by_camera),
                    source_fps_origin_by_camera=dict(origin_by_camera),
                    source_fps=episode_fps,
                    source_fps_origin=aggregate_origin,
                    observed_step=observed_step,
                    segment_index=segment_index,
                    segment_count=len(ranges),
                )
            )
            episode_index += 1
    return plans


def relative_image_root(input_dir: Path, output_dir: Path) -> str:
    """从 dataset/ 指向图片目录的相对路径，写进 info.json 的 extras.image_root。"""
    try:
        return Path(os.path.relpath(input_dir.resolve(), output_dir.resolve())).as_posix()
    except ValueError:
        # Windows 跨盘符无法相对化，退回绝对路径
        return input_dir.resolve().as_posix()


def build_info_json(
    args: argparse.Namespace,
    cameras: List[str],
    episodes: List[EpisodePlan],
    dims_by_camera: Dict[str, Counter],
    total_samples: int,
    image_root: str,
    group_by: str,
    source_root: Optional[str] = None,
    camera_names: Optional[Dict[str, str]] = None,
) -> dict:
    """组装 meta/info.json（纯清单，多相机各一个 feature）。"""
    camera_shapes: Dict[str, Tuple[int, int]] = {}
    for camera in cameras:
        dims = dims_by_camera.get(camera) or Counter()
        if dims:
            (width, height), _ = dims.most_common(1)[0]
        else:
            width, height = 0, 0
        camera_shapes[camera] = (width, height)

    origins = {episode.source_fps_origin for episode in episodes}
    if origins == {FPS_ORIGIN_RECORDED}:
        timestamp_source = "source_frame / orig_fps（精确）"
    elif FPS_ORIGIN_NOMINAL in origins:
        timestamp_source = (
            "frame_index / fps（部分视频未获得原始帧率，走名义值；"
            "该部分按保留帧数计时，帧号有缺口时会偏小）"
        )
    else:
        timestamp_source = (
            "source_frame / orig_fps（部分视频的 orig_fps 是推断值，"
            "见 source_fps_origin_by_video）"
        )

    sources = []
    for episode in episodes:
        if not episode.frames:
            continue
        camera_entries = []
        for camera in episode.cameras:
            width, height = camera_shapes.get(camera, (0, 0))
            frames = episode.camera_frames[camera]
            camera_entries.append(
                {
                    "name": camera,
                    "source_video": episode.source_prefixes[camera],
                    "orig_fps": episode.source_fps_by_camera.get(camera),
                    "orig_fps_origin": episode.source_fps_origin_by_camera.get(camera),
                    "observed_frame_step": episode.observed_step,
                    "shape": [height, width, 3],
                    "first_source_frame": frames[0].source_frame,
                    "last_source_frame": frames[-1].source_frame,
                }
            )
        sources.append(
            {
                "episode": episode.episode,
                "episode_dir": episode.episode_dir,
                "episode_index": episode.episode_index,
                "source_video": episode.source_video,
                "frames": episode.length,
                "fps": args.fps,
                "orig_fps": episode.source_fps,
                "orig_fps_origin": episode.source_fps_origin,
                "observed_frame_step": episode.observed_step,
                "first_source_frame": episode.frames[0].source_frame,
                "last_source_frame": episode.frames[-1].source_frame,
                "cameras": camera_entries,
            }
        )

    fps_by_video: Dict[str, Optional[float]] = {}
    origin_by_video: Dict[str, Optional[str]] = {}
    step_by_video: Dict[str, Optional[int]] = {}
    for episode in episodes:
        for camera in episode.cameras:
            prefix = episode.source_prefixes[camera]
            fps_by_video[prefix] = episode.source_fps_by_camera.get(camera)
            origin_by_video[prefix] = episode.source_fps_origin_by_camera.get(camera)
            step_by_video[prefix] = episode.observed_step

    features = {}
    for camera in cameras:
        width, height = camera_shapes[camera]
        features[camera_key(camera)] = {
            "dtype": "image",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            # 本项目扩展：blur_score/is_blurry 按相机顺序排列。
            "blur_score": {
                "dtype": "float32",
                "shape": [len(cameras)],
                "names": cameras if len(cameras) > 1 else None,
            },
            "is_blurry": {
                "dtype": "bool",
                "shape": [len(cameras)],
                "names": cameras if len(cameras) > 1 else None,
            },
        }
    )

    return {
        "codebase_version": "v3.0-manifest",
        "robot_type": "unknown",
        "total_episodes": len(episodes),
        "total_frames": total_samples,
        "total_tasks": 0,
        "total_videos": 0,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": features,
        "extras": {
            "note": (
                "manifest-only intermediate; frames remain in the step-1 image directory"
            ),
            "generated_from": str(args.input),
            "image_root": image_root,
            "group_by": group_by,
            "source_root": source_root,
            "camera_names": dict(camera_names or {}),
            "cameras": list(cameras),
            "camera_key": camera_key(cameras[0]) if len(cameras) == 1 else None,
            "total_camera_rows": total_samples * len(cameras),
            "source_fps_by_video": fps_by_video,
            "source_fps_origin_by_video": origin_by_video,
            "observed_frame_step_by_video": step_by_video,
            "blur_score_threshold": args.blur_threshold,
            "max_episode_seconds": args.max_episode_seconds,
            "timestamp_source": timestamp_source,
            "sources": sources,
        },
    }


def build_readme(
    args: argparse.Namespace,
    cameras: List[str],
    counters: Counters,
    episodes: List[EpisodePlan],
) -> str:
    """生成 meta/README.md：说明纯清单产物与多相机契约。"""
    lines = []
    for episode in episodes:
        for camera in episode.cameras:
            fps = episode.source_fps_by_camera.get(camera)
            fps_text = f"{fps:g}" if fps else "未知"
            origin = episode.source_fps_origin_by_camera.get(camera)
            lines.append(
                f"| {episode.episode} | {camera} | {episode.source_prefixes[camera]} | "
                f"{fps_text} | {episode.observed_step or '-'} | {origin} | {episode.length} |"
            )
    rows = "\n".join(lines)
    timestamp_note = (
        "对每个 episode 独立计算：`(source_frame - 本段首帧) / 该视频的原始帧率`，"
        "即**相对本段起点**的秒数（lerobot 写的就是 `frame_index / fps`，两者一致）。"
        "跨段的绝对位置看 `source_frame`。所有相机共用同一个 timestamp。\n\n"
        "| episode | camera | source_video | 原始帧率 | 观测步长 | 帧率来源 | 帧数 |\n"
        "|---|---|---|---|---|---|---|\n"
        f"{rows}\n\n"
        "帧率来源的含义：\n\n"
        f"- `{FPS_ORIGIN_RECORDED}`：精确值，来自 `clean_video.py` 写下的原始帧率\n"
        f"- `{FPS_ORIGIN_INFERRED}`：从帧号间隔反推的近似值，误差约 ±{args.fps / 2:g} fps"
        f"（步长 × {args.fps}）；要精确值请重跑 `clean_video.py` 生成 `{IMAGES_META_FILENAME}`\n"
        f"- `{FPS_ORIGIN_NOMINAL}`：**该 episode 没有可用的原始帧率**，"
        f"退回 `frame_index / {args.fps}`。该时间轴按保留帧数计时，帧号有缺口时会系统性偏小\n"
    )
    return f"""# 清单说明

本目录由 `format_dataset.py` 生成，**只包含清单**；帧图片仍留在第一步的图片目录
（`info.json` 的 `extras.image_root`），这里不复制、不链接图片。

## 文件

| 文件 | 内容 |
|---|---|
| `frames.jsonl` | 一行一个（相机，帧）；`path` 相对 `image_root` |
| `episodes.jsonl` | 一行一个 episode/segment |
| `info.json` | 每路相机的形状、fps、阈值、image_root、来源记录 |
| `tasks.jsonl` | 任务标注，当前为空 |
| `actions/` | 可选；由 `import_actions.py` 写入 |

最终 LeRobot 数据集由 `export_lerobot.py` 生成。本目录不是 LeRobot 数据集，
也不能被 lerobot loader 直接加载。

## 时序说明

`clean_video.py` 保存全部采样帧，帧落在 `1/fps` 的均匀网格上。一个源视频可能被
`--max-episode-seconds`（本次 {args.max_episode_seconds:g} 秒）切成多个 episode；
切分点记在 `episodes.jsonl` 的 `segment_index` / `segment_count`。

- `frame_index`：episode 内的顺序位置（0, 1, 2, ... 连续）
- `source_frame`：原始视频中的帧号（跨段不重置）
- `camera`：该行属于哪一路相机
- `blur_score` / `is_blurry`：该相机该帧的分数和阈值标记
- `path`：相对 `image_root` 的文件名
- `timestamp`：{timestamp_note}

`info.json` 的 `extras.cameras` 是相机顺序；`blur_score` / `is_blurry` 向量
（以及最终数据集里的 `blur_score`）都按这个顺序排列。所有相机必须同步：帧数、
`source_frame` 序列和原始 fps 一致。

## 扫描统计

| 项目 | 数量 |
|---|---|
| 文件名不匹配（跳过） | {counters.unparsed} |
| 重复帧号（跳过） | {counters.duplicate} |
| 损坏 / 空文件（跳过） | {counters.corrupt} |
| dir 模式下未登记的图片前缀（跳过） | {counters.unassigned} |
"""


def report_plan(episodes: List[EpisodePlan], counters: Counters, logger: logging.Logger) -> int:
    """打印扫描结果，返回同步采样数。"""
    total = 0
    logger.info("=== 扫描结果 ===")
    logger.info(
        "%-24s %8s %10s %14s %14s",
        "episode", "cameras", "samples", "first_frame", "last_frame",
    )
    for episode in episodes:
        first = episode.frames[0].source_frame
        last = episode.frames[-1].source_frame
        logger.info(
            "%-24s %8d %10d %14d %14d",
            episode.episode,
            len(episode.cameras),
            episode.length,
            first,
            last,
        )
        total += episode.length
    logger.info(
        "合计：%d 个 episode，%d 个同步采样，%d 个相机帧"
        "（跳过：不匹配 %d，重复 %d，损坏 %d，未登记 %d）",
        len(episodes),
        total,
        total * (len(episodes[0].cameras) if episodes else 0),
        counters.unparsed,
        counters.duplicate,
        counters.corrupt,
        counters.unassigned,
    )
    return total


def report_blur(
    episodes: List[EpisodePlan], threshold: float, logger: logging.Logger
) -> None:
    """报告按阈值会被标为模糊的相机帧数。只报告，不删任何帧。"""
    logger.info("=== 模糊标记（阈值 %.1f，只标记不删帧）===", threshold)
    logger.info("%-24s %8s %10s %10s %8s", "episode", "cameras", "samples", "cam_rows", "blurry")
    total = blurry_total = unknown_total = 0
    for episode in episodes:
        camera_rows = sum(len(frames) for frames in episode.camera_frames.values())
        blurry = sum(
            1
            for frames in episode.camera_frames.values()
            for frame in frames
            if frame.is_blurry
        )
        unknown_total += sum(
            1
            for frames in episode.camera_frames.values()
            for frame in frames
            if frame.blur_score is None
        )
        total += camera_rows
        blurry_total += blurry
        rate = f"{blurry / camera_rows * 100:.1f}%" if camera_rows else "-"
        logger.info(
            "%-24s %8d %10d %10d %8s",
            episode.episode,
            len(episode.cameras),
            episode.length,
            camera_rows,
            rate,
        )

    if total:
        logger.info(
            "合计：%d 个相机帧中 %d 个标为模糊（%.1f%%）",
            total,
            blurry_total,
            blurry_total / total * 100,
        )
    if unknown_total:
        logger.warning(
            "另有 %d 个相机帧没有模糊分数（缺 _scores.jsonl 或无对应记录），其 is_blurry 为 null",
            unknown_total,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把抽取出的图片整理成清单（中间产物，仅用标准库）。"
    )
    parser.add_argument("--input", "-i", default="output/images", help="输入图片目录（默认 output/images）")
    parser.add_argument("--output", "-o", default="dataset", help="输出清单目录（默认 dataset）")
    parser.add_argument("--fps", type=int, default=5, help="处理帧率，写入 info.json 的 fps 字段（默认 5）")
    parser.add_argument(
        "--group-by",
        choices=["file", "dir"],
        default=None,
        help="分组方式；缺省沿用 _meta.json（旧产物/缺省为 file）",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help=f"file 模式的相机名（默认 {DEFAULT_CAMERA}）；dir 模式请用 --camera-names",
    )
    parser.add_argument(
        "--camera-names",
        default=None,
        help="可选 JSON：文件名 stem -> 相机名；可在不重跑第一步的情况下改名",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Laplacian 方差低于此值的帧在清单里标为模糊。只做标记，不删除任何帧（默认 100）",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=60.0,
        help="单个 episode 的时长上限（秒）；超长的视频按此切成多段，0 表示不切（默认 60）",
    )
    parser.add_argument("--force", action="store_true", help="已存在清单时覆盖")
    parser.add_argument("--dry-run", action="store_true", help="只扫描并打印报告，不创建任何文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("format_dataset")

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.fps <= 0:
        logger.error("--fps 必须为正整数")
        return 1
    if not input_dir.is_dir():
        logger.error("输入目录不存在或不是目录：%s", input_dir)
        return 1

    info_path = output_dir / META_DIRNAME / INFO_JSON
    if info_path.exists() and not args.force and not args.dry_run:
        logger.error("输出目录已存在数据集（%s）。如需重新生成，请加 --force", info_path)
        return 1

    counters = Counters()
    start_ts = time.time()

    images_meta = load_images_meta(input_dir)
    meta_fps = extract_source_fps(images_meta)
    meta_group_by = images_meta.get("group_by") if images_meta.get("format") == 3 else "file"
    actual_group_by = args.group_by or meta_group_by or "file"
    if actual_group_by == "dir" and args.camera is not None:
        logger.error("dir 模式不能同时传 --camera；相机名请用 --camera-names 或文件名")
        return 1
    try:
        camera_names = load_camera_names(
            Path(args.camera_names) if args.camera_names else None
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    default_camera = args.camera or DEFAULT_CAMERA

    recorded_fps = images_meta.get("target_fps")
    if isinstance(recorded_fps, (int, float)) and recorded_fps != args.fps:
        logger.warning(
            "%s 记录的处理帧率是 %s，与本次 --fps %d 不一致，请确认是否为同一批数据",
            images_meta.get("tool", "上游"),
            recorded_fps,
            args.fps,
        )

    blur_scores = load_blur_scores(input_dir)

    try:
        max_frames = int(round(args.max_episode_seconds * args.fps))
    except (TypeError, ValueError):
        max_frames = 0
    if args.max_episode_seconds <= 0 or max_frames <= 0:
        max_frames = 0
        logger.info("--max-episode-seconds 为 0：长视频不切分，一个视频就是一个 episode")
    else:
        logger.info(
            "--max-episode-seconds %.1f：每 %d 帧切一段（%.1f 秒 @ %d fps）",
            args.max_episode_seconds,
            max_frames,
            max_frames / args.fps,
            args.fps,
        )

    try:
        episodes = scan_input(
            input_dir,
            counters,
            args.fps,
            blur_scores,
            args.blur_threshold,
            max_frames,
            group_by=args.group_by,
            default_camera=default_camera,
            images_meta=images_meta,
            camera_names=camera_names,
            meta_fps=meta_fps,
        )
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    episodes = [episode for episode in episodes if episode.frames]
    if not episodes:
        logger.error("在 %s 中没有找到任何可用的帧图片", input_dir)
        return 1

    total_samples = report_plan(episodes, counters, logger)
    cameras = episodes[0].cameras
    logger.info("=== 原始帧率 ===")
    logger.info(
        "%-24s %-12s %10s %12s %12s %10s",
        "episode", "camera", "orig_fps", "step(观测)", "step(期望)", "来源",
    )
    for episode in episodes:
        for camera in episode.cameras:
            fps = episode.source_fps_by_camera.get(camera)
            expected = max(1, int(round(fps / float(args.fps)))) if fps else None
            logger.info(
                "%-24s %-12s %10s %12s %12s %10s",
                episode.episode,
                camera,
                f"{fps:g}" if fps else "未知",
                episode.observed_step if episode.observed_step else "-",
                expected if expected else "-",
                episode.source_fps_origin_by_camera.get(camera),
            )
    distinct_steps = {e.observed_step for e in episodes if e.observed_step}
    if len(distinct_steps) > 1:
        logger.info(
            "注意：各视频的抽帧步长不同（%s），说明这批素材原始帧率不统一",
            ", ".join(str(s) for s in sorted(distinct_steps)),
        )

    report_blur(episodes, args.blur_threshold, logger)

    if args.dry_run:
        logger.info("--dry-run：未创建任何文件，耗时 %.2fs", time.time() - start_ts)
        return 0

    # ---- 建清单目录 ----
    meta_dir = output_dir / META_DIRNAME
    try:
        meta_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("创建输出目录失败：%s", output_dir)
        return 1

    # ---- 收集元信息（不复制/链接图片）----
    dims_by_camera: Dict[str, Counter] = {camera: Counter() for camera in cameras}
    episode_rows: List[EpisodeRow] = []
    frame_rows: List[FrameRow] = []
    global_index = 0

    for episode in episodes:
        # timestamp 相对本段起点计时，而不是视频起点；所有相机共用。
        segment_start = episode.frames[0].source_frame
        timestamps = []
        for position, entry in enumerate(episode.frames):
            if episode.source_fps:
                timestamps.append(
                    round((entry.source_frame - segment_start) / episode.source_fps, 6)
                )
            else:
                timestamps.append(round(position / args.fps, 6))

        for camera in episode.cameras:
            for position, entry in enumerate(episode.camera_frames[camera]):
                dims_by_camera[camera][(entry.width, entry.height)] += 1
                frame_rows.append(
                    {
                        "index": global_index,
                        "episode_index": episode.episode_index,
                        "frame_index": position,
                        "source_frame": entry.source_frame,
                        "camera": camera,
                        # 模糊分数是上游算好的原始值；is_blurry 是本次阈值的结论。
                        "blur_score": entry.blur_score,
                        "is_blurry": entry.is_blurry,
                        "timestamp": timestamps[position],
                        "path": Path(os.path.relpath(entry.src, input_dir)).as_posix(),
                    }
                )
                global_index += 1

        episode_rows.append(
            {
                "episode_index": episode.episode_index,
                "tasks": [],
                "length": episode.length,
                "episode": episode.episode,
                "source_video": episode.source_video,
                "cameras": list(episode.cameras),
                "source_videos": dict(episode.source_prefixes),
                "fps": args.fps,
                "source_fps": episode.source_fps,
                "source_fps_origin": episode.source_fps_origin,
                "source_fps_by_camera": dict(episode.source_fps_by_camera),
                "source_fps_origin_by_camera": dict(episode.source_fps_origin_by_camera),
                "first_source_frame": episode.frames[0].source_frame,
                "last_source_frame": episode.frames[-1].source_frame,
                "segment_index": episode.segment_index,
                "segment_count": episode.segment_count,
            }
        )
        logger.info(
            "episode %d（%s，%d 路相机）完成：%d 帧/相机",
            episode.episode_index,
            episode.episode,
            len(episode.cameras),
            episode.length,
        )

    if not episode_rows:
        logger.error("所有 episode 都没有可用帧，未生成数据集")
        return 1

    # ---- 写 meta/ ----
    image_root = relative_image_root(input_dir, output_dir)
    try:
        write_json(
            info_path,
            build_info_json(
                args,
                cameras,
                episodes,
                dims_by_camera,
                total_samples,
                image_root,
                actual_group_by,
                source_root=images_meta.get("source_root"),
                camera_names={
                    **(images_meta.get("camera_names") or {}),
                    **camera_names,
                },
            ),
        )
        write_jsonl(meta_dir / EPISODES_JSONL, episode_rows)
        write_jsonl(meta_dir / FRAMES_JSONL, frame_rows)
        # 没有任务标注，仍然生成空文件以保住结构
        write_jsonl(meta_dir / TASKS_JSONL, [])
        (meta_dir / "README.md").write_text(
            build_readme(args, cameras, counters, episodes), encoding="utf-8"
        )
    except OSError:
        logger.exception("写入 meta 文件失败：%s", meta_dir)
        return 1

    for camera, dims in dims_by_camera.items():
        if len(dims) > 1:
            top = ", ".join(f"{w}x{h}×{n}" for (w, h), n in dims.most_common(5))
            logger.warning(
                "相机 %s 检测到多种图片尺寸，info.json 取众数；分布：%s",
                camera,
                top,
            )

    logger.info("=== 完成 ===")
    logger.info(
        "episodes=%d samples=%d camera_rows=%d cameras=%s image_root=%s",
        len(episode_rows),
        total_samples,
        len(frame_rows),
        ", ".join(cameras),
        image_root,
    )
    logger.info("清单目录：%s（耗时 %.2fs）", output_dir.resolve(), time.time() - start_ts)
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
