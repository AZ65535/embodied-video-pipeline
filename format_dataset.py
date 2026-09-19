#!/usr/bin/env python
"""把 clean_video.py 抽出的清晰帧整理成仿 LeRobot v2.0 布局的数据集骨架。

仅依赖 Python 标准库（未引入 pyarrow / lerobot / 任何图像库）。

产物与上游 LeRobot v2.0 的差异、以及本脚本做不到的事情，
全部记录在生成出来的 meta/README.md 里，请以那份文档为准。
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 形如 "1_frame_000318.jpg"：视频名用贪婪匹配，取最后一个 "_frame_" 作为分隔，
# 这样视频名本身含下划线、甚至含 "_frame_" 也能正确切分。
FRAME_NAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)

# JPEG SOF 标记：C0-CF 里除 C4(DHT)/C8(JPG)/CC(DAC) 之外都是帧头，携带宽高。
SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
# 无长度字段的独立标记：TEM / RST0-7 / SOI / EOI
STANDALONE_MARKERS = frozenset({0x01, 0xD8, 0xD9, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7})

# Windows 传统 MAX_PATH 上限；超出时部分 API 会莫名失败，提前告警。
WINDOWS_PATH_WARN_LENGTH = 250

DEFAULT_CAMERA = "front"

# clean_video.py 写在图片目录里的处理参数与原始帧率记录。
IMAGES_META_FILENAME = "_meta.json"

# 原始帧率的三种来源，会写进 info.json / README，方便判断 timestamp 可不可信。
FPS_ORIGIN_RECORDED = "_meta.json"           # 精确
FPS_ORIGIN_INFERRED = "inferred_from_frame_gaps"  # 近似，误差约 ±fps/2
FPS_ORIGIN_NOMINAL = "nominal"               # 兜底，时间轴会偏小


@dataclass
class FrameEntry:
    """输入目录里的一帧原图。宽高在扫描阶段读出，避免落盘前重复读文件。"""

    src: Path
    source_frame: int
    width: int
    height: int


@dataclass
class EpisodePlan:
    """一个源视频对应的 episode（尚未落盘）。"""

    episode_index: int
    source_video: str
    frames: List[FrameEntry] = field(default_factory=list)
    # 该视频的原始帧率；None 表示未知（timestamp 退回名义值）
    source_fps: Optional[float] = None
    # 帧率的来源，取值见 FPS_ORIGIN_* 常量
    source_fps_origin: Optional[str] = None
    # 从原始帧号的间隔反推出的抽帧步长
    observed_step: Optional[int] = None


@dataclass
class Counters:
    """扫描阶段的异常计数，最后汇总打印。"""

    unparsed: int = 0
    duplicate: int = 0
    corrupt: int = 0
    copied: int = 0
    linked: int = 0
    already_present: int = 0


def natural_key(text: str) -> List[object]:
    """自然排序键：让 "2" 排在 "10" 前面，而不是按字典序倒过来。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def parse_frame_name(path: Path) -> Optional[Tuple[str, int]]:
    """从文件名解析出 (视频名, 原始帧号)；不匹配返回 None。"""
    match = FRAME_NAME_RE.match(path.name)
    if match is None:
        return None
    return match.group("video"), int(match.group("frame"))


def probe_jpeg(path: Path) -> Optional[Tuple[int, int]]:
    """只读 JPEG 头部，返回 (width, height)。

    顺带充当完整性校验：结构不对（缺 SOI、段长非法、SOS 之前没有 SOF）
    一律返回 None，调用方据此跳过并计入损坏帧。
    """
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"\xff\xd8":
                return None  # 缺 SOI

            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None

                code = marker[1]
                # 允许 FF FF ... xx 形式的填充字节
                while code == 0xFF:
                    extra = handle.read(1)
                    if not extra:
                        return None
                    code = extra[0]

                if code == 0xDA or code == 0xD9:
                    return None  # 遇到 SOS/EOI 还没见到 SOF，说明不是常规 JPEG
                if code in STANDALONE_MARKERS:
                    continue

                length_bytes = handle.read(2)
                if len(length_bytes) < 2:
                    return None
                segment_length = int.from_bytes(length_bytes, "big")
                if segment_length < 2:
                    return None

                if code in SOF_MARKERS:
                    body = handle.read(5)  # 精度(1) + 高(2) + 宽(2)
                    if len(body) < 5:
                        return None
                    height = int.from_bytes(body[1:3], "big")
                    width = int.from_bytes(body[3:5], "big")
                    if width <= 0 or height <= 0:
                        return None
                    return width, height

                handle.seek(segment_length - 2, os.SEEK_CUR)
    except OSError:
        return None


def has_eoi(path: Path) -> bool:
    """检查文件末尾是否有 JPEG 结束标记 EOI（FFD9）。"""
    try:
        size = path.stat().st_size
        if size < 4:
            return False
        with path.open("rb") as handle:
            handle.seek(-2, os.SEEK_END)
            return handle.read(2) == b"\xff\xd9"
    except OSError:
        return False


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


def infer_frame_step(frames: List[FrameEntry]) -> Optional[int]:
    """从原始帧号的间隔反推抽帧步长。

    clean_video.py 按固定步长采样，所以相邻保留帧的帧号差一定是步长的整数倍；
    对全部差值取 gcd 即可还原步长 —— 被丢掉的模糊帧只会把差值抬成步长的倍数，
    但只要存在任意一对相邻保留帧没被丢，gcd 就等于步长本身。
    """
    if len(frames) < 2:
        return None
    step = 0
    for prev, cur in zip(frames, frames[1:]):
        step = math.gcd(step, cur.source_frame - prev.source_frame)
        if step == 1:
            break  # gcd 不可能再变小
    return step or None


def resolve_episode_fps(
    episode: EpisodePlan, args: argparse.Namespace, meta_fps: Dict[str, float]
) -> Tuple[Optional[float], str]:
    """决定该 episode 的原始帧率，返回 (fps, 来源)。

    优先级：_meta.json 记录 > 从帧号间隔反推 > 无（用名义时间戳）。
    同时交叉校验：_meta.json 记的帧率对应的步长，应该等于从帧号反推的步长。
    """
    logger = logging.getLogger("format_dataset")
    episode.observed_step = infer_frame_step(episode.frames)

    recorded = meta_fps.get(episode.source_video)
    if recorded:
        expected_step = max(1, int(round(recorded / float(args.fps))))
        if episode.observed_step and episode.observed_step != expected_step:
            logger.warning(
                "%s：%s 记录的帧率 %g 对应步长 %d，但从帧号反推的步长是 %d，两者不符"
                "（素材与 %s 可能对不上），仍按记录值处理",
                episode.source_video,
                IMAGES_META_FILENAME,
                recorded,
                expected_step,
                episode.observed_step,
                IMAGES_META_FILENAME,
            )
        return recorded, FPS_ORIGIN_RECORDED

    if episode.observed_step:
        inferred = episode.observed_step * float(args.fps)
        logger.warning(
            "%s：%s 中没有该视频的帧率记录，按帧号间隔反推为 %g fps（步长 %d × %d）。"
            "这是近似值（误差约 ±%g fps）；要精确值请重跑 clean_video.py 生成 %s",
            episode.source_video,
            IMAGES_META_FILENAME,
            inferred,
            episode.observed_step,
            args.fps,
            args.fps / 2.0,
            IMAGES_META_FILENAME,
        )
        return inferred, FPS_ORIGIN_INFERRED

    logger.warning(
        "%s：既无 %s 记录、也无法从帧号反推帧率（只有 %d 帧），"
        "将使用名义时间戳 frame_index/%d —— 该时间轴忽略了被丢弃的模糊帧，会偏小",
        episode.source_video,
        IMAGES_META_FILENAME,
        len(episode.frames),
        args.fps,
    )
    return None, FPS_ORIGIN_NOMINAL


def scan_input(input_dir: Path, counters: Counters) -> List[EpisodePlan]:
    """扫描输入目录，按视频名前缀分组。"""
    logger = logging.getLogger("format_dataset")
    grouped: Dict[str, Dict[int, Tuple[Path, int, int]]] = defaultdict(dict)

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == IMAGES_META_FILENAME:
            continue  # 由 load_images_meta 单独读取，不是帧图片

        parsed = parse_frame_name(path)
        if parsed is None:
            logger.warning("跳过：文件名不符合 {视频名}_frame_{帧号}.jpg 规范 -> %s", path.name)
            counters.unparsed += 1
            continue

        video_name, source_frame = parsed
        if source_frame in grouped[video_name]:
            # 同名同帧号出现两次（例如历史上两轮 glob 把同一文件收了两次）
            logger.warning(
                "跳过重复帧：%s 的帧号 %d 已由 %s 提供，忽略 %s",
                video_name,
                source_frame,
                grouped[video_name][source_frame].name,
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

    # 视频名自然序；组内按原始帧号整数序（不是字符串序）
    plans: List[EpisodePlan] = []
    for episode_index, video_name in enumerate(sorted(grouped, key=natural_key)):
        frames = [
            FrameEntry(src=path, source_frame=frame_no, width=width, height=height)
            for frame_no, (path, width, height) in sorted(grouped[video_name].items())
        ]
        plans.append(EpisodePlan(episode_index=episode_index, source_video=video_name, frames=frames))
    return plans


def resolve_link_mode(requested: str, sample_src: Path, dst_dir: Path) -> str:
    """把 auto 解析成实际可用的方式：硬链接 -> 软链接 -> 复制。"""
    logger = logging.getLogger("format_dataset")
    if requested != "auto":
        return requested

    probe_dst = dst_dir / ".__link_probe__"
    for mode in ("hardlink", "symlink"):
        try:
            if probe_dst.exists() or probe_dst.is_symlink():
                probe_dst.unlink()
            create_link(mode, sample_src.resolve(), probe_dst)
        except OSError as exc:
            logger.debug("%s 探测失败：%s", mode, exc)
            continue
        finally:
            try:
                if probe_dst.exists() or probe_dst.is_symlink():
                    probe_dst.unlink()
            except OSError:
                pass
        logger.info("链接方式自动判定为：%s", mode)
        return mode

    logger.info("链接方式自动判定为：copy（硬链接与软链接均不可用）")
    return "copy"


def create_link(mode: str, src: Path, dst: Path) -> None:
    """按指定方式把 src 落成 dst。"""
    if mode == "hardlink":
        os.link(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def materialize(
    src: Path, dst: Path, mode: str, force: bool, counters: Counters
) -> Optional[str]:
    """把一帧原图落到目标路径。返回实际动作，失败返回 None。"""
    logger = logging.getLogger("format_dataset")
    try:
        if dst.exists() or dst.is_symlink():
            try:
                if dst.samefile(src):
                    counters.already_present += 1
                    return "present"
            except OSError:
                pass
            if not force:
                logger.warning("目标已存在且不是同一文件，跳过（用 --force 覆盖）：%s", dst)
                return None
            dst.unlink()

        create_link(mode, src.resolve(), dst)
    except OSError:
        logger.exception("落盘失败：%s -> %s", src, dst)
        return None

    if mode == "copy":
        counters.copied += 1
    else:
        counters.linked += 1
    return mode


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def build_info_json(
    args: argparse.Namespace,
    camera_key: str,
    episodes: List[EpisodePlan],
    dims: Counter,
    total_frames: int,
) -> dict:
    """组装 meta/info.json。标准字段对齐上游，非标准信息一律塞进 extras。"""
    if dims:
        (width, height), _ = dims.most_common(1)[0]
    else:
        width, height = 0, 0

    origins = {episode.source_fps_origin for episode in episodes}
    if origins == {FPS_ORIGIN_RECORDED}:
        timestamp_source = "source_frame / orig_fps（精确，含模糊帧造成的空洞）"
    elif FPS_ORIGIN_NOMINAL in origins:
        timestamp_source = (
            "frame_index / fps（部分视频未获得原始帧率，走名义值；"
            "该部分时间轴忽略模糊帧空洞，会偏小）"
        )
    else:
        timestamp_source = (
            "source_frame / orig_fps（部分视频的 orig_fps 是推断值，"
            "见 source_fps_origin_by_video）"
        )

    sources = [
        {
            "source_video": episode.source_video,
            "episode_index": episode.episode_index,
            "frames": len(episode.frames),
            "fps": args.fps,
            "orig_fps": episode.source_fps,
            "orig_fps_origin": episode.source_fps_origin,
            "observed_frame_step": episode.observed_step,
            "first_source_frame": episode.frames[0].source_frame,
            "last_source_frame": episode.frames[-1].source_frame,
        }
        for episode in episodes
        if episode.frames
    ]

    return {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": max(1, -(-len(episodes) // args.chunks_size)) if episodes else 0,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            camera_key: {
                # 上游此处为 "video"；我们存的是 jpg 帧序列，故标 "image"。
                "dtype": "image",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "frame_dir": f"videos/chunk-{{episode_chunk:03d}}/{{video_key}}/episode_{{episode_index:06d}}/"
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "extras": {
            "note": (
                "images-only skeleton mimicking LeRobot v2.0 layout; "
                "no action/state, no parquet, no mp4"
            ),
            "generated_from": str(args.input),
            "camera_key": camera_key,
            "link_mode": args.link_mode,
            "source_fps_by_video": {
                episode.source_video: episode.source_fps for episode in episodes
            },
            "source_fps_origin_by_video": {
                episode.source_video: episode.source_fps_origin for episode in episodes
            },
            "observed_frame_step_by_video": {
                episode.source_video: episode.observed_step for episode in episodes
            },
            "timestamp_source": timestamp_source,
            "sources": sources,
        },
    }


def build_readme(
    args: argparse.Namespace,
    camera_key: str,
    counters: Counters,
    episodes: List[EpisodePlan],
) -> str:
    """生成 meta/README.md：把与上游 LeRobot 的差异讲清楚。"""
    lines = []
    for episode in episodes:
        fps_text = f"{episode.source_fps:g}" if episode.source_fps else "未知"
        lines.append(
            f"| {episode.source_video} | {fps_text} | {episode.observed_step or '-'} | "
            f"{episode.source_fps_origin} | {len(episode.frames)} |"
        )
    rows = "\n".join(lines)
    timestamp_note = (
        "对每个 episode 独立计算：`source_frame / 该视频的原始帧率`。\n\n"
        "| source_video | 原始帧率 | 观测步长 | 帧率来源 | 帧数 |\n"
        "|---|---|---|---|---|\n"
        f"{rows}\n\n"
        "帧率来源的含义：\n\n"
        f"- `{FPS_ORIGIN_RECORDED}`：精确值，来自 `clean_video.py` 写下的原始帧率\n"
        f"- `{FPS_ORIGIN_INFERRED}`：从帧号间隔反推的近似值，误差约 ±{args.fps / 2:g} fps"
        f"（步长 × {args.fps}）；要精确值请重跑 `clean_video.py` 生成 `{IMAGES_META_FILENAME}`\n"
        f"- `{FPS_ORIGIN_NOMINAL}`：**该 episode 没有可用的原始帧率**，"
        f"退回 `frame_index / {args.fps}`。该时间轴忽略模糊帧造成的空洞，会系统性偏小\n"
    )
    return f"""# 数据集说明（仿 LeRobot v2.0）

本目录由 `format_dataset.py` 生成，**仅使用 Python 标准库**，未引入 `lerobot` / `pyarrow`。

## 与上游 LeRobot v2.0 的差异

| 项目 | 上游 LeRobot v2.0 | 本数据集 |
|---|---|---|
| `data/` | 每个 episode 一个 `.parquet`（含 `observation.state` / `action` / `timestamp` 等） | **空目录**，保留结构占位 |
| `videos/` | 每个 episode 一个 `.mp4` | 每个 episode 一个**图片目录**，内含逐帧 `.jpg` |
| 相机特征 `dtype` | `"video"` | `"image"` |
| `meta/episodes.jsonl` | 有 | 有（语义一致，一行一个 episode） |
| `meta/frames.jsonl` | **无此文件** | 有，本脚本扩展，一行一帧 |
| `meta/tasks.jsonl` | 有 | 有，但无任务标注时为空文件 |
| `meta/stats.json` | 有 | **未生成**（需要解码像素才能统计） |

### 最根本的差异

LeRobot 是**模仿学习**数据集，核心是 `(observation, action)` 配对。
本数据集**只有 observation，没有 action / state / reward**，
因此它可以用于感知预训练、世界模型、视频预测，但**不能直接用于训练策略**，
也无法被 `lerobot` 的 loader 直接加载。

## 时序说明

`clean_video.py` 抽帧时丢弃了模糊帧，所以帧序列存在**时序空洞**。
文件名已按 episode 内位置重命名为 `frame_000000.jpg`，
原始帧号保存在 `meta/frames.jsonl` 的 `source_frame` 字段中。

- `frame_index`：episode 内的顺序位置（0, 1, 2, ... 连续）
- `source_frame`：原始视频中的帧号（**不连续**，跳过的就是被判定为模糊的帧）
- `timestamp`：{timestamp_note}

## 文件落盘方式

`--link-mode` 实际使用：**{args.link_mode}**

采用硬链接时，本目录的图片与原始 `output/images/` 是**同一个文件**（共享 inode），
不额外占用磁盘空间；但修改其中任何一方都会影响另一方。
若需完全独立，请改用 `--link-mode copy` 重新生成。

## 未生成的字段

`meta/info.json` 中的 `total_videos` 恒为 `0`，因为我们没有生成 mp4 文件。

## 扫描统计

| 项目 | 数量 |
|---|---|
| 已存在（跳过，未重复落盘） | {counters.already_present} |
| 新建链接 | {counters.linked} |
| 复制 | {counters.copied} |
| 文件名不匹配（跳过） | {counters.unparsed} |
| 重复帧号（跳过） | {counters.duplicate} |
| 损坏 / 空文件（跳过） | {counters.corrupt} |
"""


def report_plan(episodes: List[EpisodePlan], counters: Counters, logger: logging.Logger) -> int:
    """打印扫描结果，返回总帧数。"""
    total = 0
    logger.info("=== 扫描结果 ===")
    logger.info("%-30s %10s %14s %14s", "source_video", "frames", "first_frame", "last_frame")
    for episode in episodes:
        first = episode.frames[0].source_frame
        last = episode.frames[-1].source_frame
        logger.info(
            "%-30s %10d %14d %14d", episode.source_video, len(episode.frames), first, last
        )
        total += len(episode.frames)
    logger.info(
        "合计：%d 个 episode，%d 帧（跳过：不匹配 %d，重复 %d，损坏 %d）",
        len(episodes),
        total,
        counters.unparsed,
        counters.duplicate,
        counters.corrupt,
    )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把抽取出的图片整理成仿 LeRobot v2.0 布局的数据集骨架（仅用标准库）。"
    )
    parser.add_argument("--input", "-i", default="output/images", help="输入图片目录（默认 output/images）")
    parser.add_argument("--output", "-o", default="dataset", help="输出数据集目录（默认 dataset）")
    parser.add_argument("--fps", type=int, default=5, help="处理帧率，写入 info.json 的 fps 字段（默认 5）")
    parser.add_argument(
        "--camera", default=DEFAULT_CAMERA, help=f"相机名，生成 observation.images.<camera>（默认 {DEFAULT_CAMERA}）"
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="文件落盘方式，auto = 硬链接 -> 软链接 -> 复制 依次降级（默认 auto）",
    )
    parser.add_argument("--chunks-size", type=int, default=1000, help="每个 chunk 容纳的 episode 数（默认 1000）")
    parser.add_argument("--force", action="store_true", help="输出目录已存在数据集时覆盖，并覆盖同名目标文件")
    parser.add_argument("--dry-run", action="store_true", help="只扫描并打印报告，不创建任何文件")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("format_dataset")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    camera_key = f"observation.images.{args.camera}"

    if args.chunks_size <= 0:
        logger.error("--chunks-size 必须为正整数")
        return 1
    if args.fps <= 0:
        logger.error("--fps 必须为正整数")
        return 1
    if not input_dir.is_dir():
        logger.error("输入目录不存在或不是目录：%s", input_dir)
        return 1

    info_path = output_dir / "meta" / "info.json"
    if info_path.exists() and not args.force and not args.dry_run:
        logger.error("输出目录已存在数据集（%s）。如需重新生成，请加 --force", info_path)
        return 1

    counters = Counters()
    start_ts = time.time()

    images_meta = load_images_meta(input_dir)
    meta_fps = extract_source_fps(images_meta)
    recorded_fps = images_meta.get("target_fps")
    if isinstance(recorded_fps, (int, float)) and recorded_fps != args.fps:
        logger.warning(
            "%s 记录的处理帧率是 %s，与本次 --fps %d 不一致，请确认是否为同一批数据",
            images_meta.get("tool", "上游"),
            recorded_fps,
            args.fps,
        )

    try:
        episodes = scan_input(input_dir, counters)
    except OSError:
        logger.exception("扫描输入目录失败：%s", input_dir)
        return 1

    episodes = [episode for episode in episodes if episode.frames]
    if not episodes:
        logger.error("在 %s 中没有找到任何可用的帧图片", input_dir)
        return 1

    for episode in episodes:
        episode.source_fps, episode.source_fps_origin = resolve_episode_fps(
            episode, args, meta_fps
        )

    total_frames = report_plan(episodes, counters, logger)
    logger.info("=== 原始帧率 ===")
    logger.info(
        "%-14s %10s %12s %12s %10s", "source_video", "orig_fps", "step(观测)", "step(期望)", "来源"
    )
    for episode in episodes:
        expected = (
            max(1, int(round(episode.source_fps / float(args.fps))))
            if episode.source_fps
            else None
        )
        logger.info(
            "%-14s %10s %12s %12s %10s",
            episode.source_video,
            f"{episode.source_fps:g}" if episode.source_fps else "未知",
            episode.observed_step if episode.observed_step else "-",
            expected if expected else "-",
            episode.source_fps_origin,
        )
    distinct_steps = {e.observed_step for e in episodes if e.observed_step}
    if len(distinct_steps) > 1:
        logger.info(
            "注意：各视频的抽帧步长不同（%s），说明这批素材原始帧率不统一",
            ", ".join(str(s) for s in sorted(distinct_steps)),
        )

    if args.dry_run:
        logger.info("--dry-run：未创建任何文件，耗时 %.2fs", time.time() - start_ts)
        return 0

    # ---- 建目录 ----
    try:
        data_dir = output_dir / "data"
        meta_dir = output_dir / "meta"
        videos_dir = output_dir / "videos"
        for directory in (data_dir, meta_dir, videos_dir):
            directory.mkdir(parents=True, exist_ok=True)
        # data/ 目前没有 parquet 可放，留 .gitkeep 保住目录结构
        (data_dir / ".gitkeep").touch()
    except OSError:
        logger.exception("创建输出目录失败：%s", output_dir)
        return 1

    # ---- 决定落盘方式 ----
    try:
        link_mode = resolve_link_mode(args.link_mode, episodes[0].frames[0].src, output_dir)
    except OSError:
        logger.exception("无法确定落盘方式，请显式指定 --link-mode copy")
        return 1
    if link_mode != args.link_mode:
        args.link_mode = link_mode

    # ---- 落盘 + 收集元信息 ----
    dims: Counter = Counter()
    episode_rows: List[dict] = []
    frame_rows: List[dict] = []
    global_index = 0
    dumped = 0
    path_warned = False
    warn_length = len(str((output_dir / "videos").resolve())) + 70

    for episode in episodes:
        chunk = episode_chunk(episode.episode_index, args.chunks_size)
        frame_dir = videos_dir / f"chunk-{chunk:03d}" / camera_key / f"episode_{episode.episode_index:06d}"

        try:
            frame_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("创建 episode 目录失败：%s", frame_dir)
            continue

        if not path_warned and warn_length > WINDOWS_PATH_WARN_LENGTH:
            logger.warning(
                "目标路径接近 Windows MAX_PATH 上限（约 %d 字符），若出现莫名 IO 失败请缩短 --output 路径",
                warn_length,
            )
            path_warned = True

        # 显式相对 output_dir 取路径：直接用 frame_dir.as_posix() 的话，
        # 一旦 --output 传的是绝对路径，写进 jsonl 的就会是本机绝对路径，数据集不可移植。
        frame_dir_rel = frame_dir.relative_to(output_dir).as_posix()
        kept = 0
        for position, entry in enumerate(episode.frames):
            dims[(entry.width, entry.height)] += 1

            if episode.source_fps:
                timestamp = round(entry.source_frame / episode.source_fps, 6)
            else:
                timestamp = round(position / args.fps, 6)

            dst = frame_dir / f"frame_{position:06d}.jpg"
            action = materialize(entry.src, dst, link_mode, args.force, counters)
            if action is None:
                continue

            frame_rows.append(
                {
                    "index": global_index,
                    "episode_index": episode.episode_index,
                    "frame_index": position,
                    "source_frame": entry.source_frame,
                    "timestamp": timestamp,
                    "path": f"{frame_dir_rel}/frame_{position:06d}.jpg",
                }
            )
            global_index += 1
            kept += 1
            dumped += 1
            if dumped % 500 == 0:
                logger.info("已落盘 %d / %d 帧", dumped, total_frames)

        if kept == 0:
            logger.warning("episode %d 没有任何可用帧，跳过", episode.episode_index)
            continue

        episode_rows.append(
            {
                "episode_index": episode.episode_index,
                "tasks": [],
                "length": kept,
                # 以下为扩展字段
                "source_video": episode.source_video,
                "frame_dir": frame_dir_rel,
                "fps": args.fps,
                "source_fps": episode.source_fps,
                "source_fps_origin": episode.source_fps_origin,
                "first_source_frame": episode.frames[0].source_frame,
                "last_source_frame": episode.frames[-1].source_frame,
            }
        )
        logger.info(
            "episode %d（%s）完成：%d 帧 -> %s",
            episode.episode_index,
            episode.source_video,
            kept,
            frame_dir_rel,
        )

    if not episode_rows:
        logger.error("所有 episode 都没有可用帧，未生成数据集")
        return 1

    # ---- 写 meta/ ----
    try:
        known = {row["episode_index"] for row in episode_rows}
        episodes = [episode for episode in episodes if episode.episode_index in known]

        write_json(
            info_path,
            build_info_json(args, camera_key, episodes, dims, len(frame_rows)),
        )
        write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
        write_jsonl(meta_dir / "frames.jsonl", frame_rows)
        # 没有任务标注，仍然生成空文件以保住结构
        write_jsonl(meta_dir / "tasks.jsonl", [])
        (meta_dir / "README.md").write_text(
            build_readme(args, camera_key, counters, episodes), encoding="utf-8"
        )
    except OSError:
        logger.exception("写入 meta 文件失败：%s", meta_dir)
        return 1

    if len(dims) > 1:
        top = ", ".join(f"{w}x{h}×{n}" for (w, h), n in dims.most_common(5))
        logger.warning("检测到多种图片尺寸，info.json 取众数；分布：%s", top)

    logger.info("=== 完成 ===")
    logger.info(
        "episodes=%d frames=%d 落盘方式=%s 链接=%d 复制=%d 已存在=%d",
        len(episode_rows),
        len(frame_rows),
        link_mode,
        counters.linked,
        counters.copied,
        counters.already_present,
    )
    logger.info("输出目录：%s（耗时 %.2fs）", output_dir.resolve(), time.time() - start_ts)
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
