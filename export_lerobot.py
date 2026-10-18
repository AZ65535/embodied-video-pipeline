#!/usr/bin/env python
"""把 format_dataset.py 产出的帧清单组装成 lerobot v3.0 格式的数据集。

管线第三步：

    dataset/ ──export_lerobot.py──▶ lerobot_dataset/

meta/ 下的 parquet 和 info.json 全部由 lerobot 自己生成，格式正确性由官方代码保证。
与上游的差异、以及本脚本做不到的事情，见 README 的「第三步」。
"""

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline_common import (
    ACTIONS_DIRNAME,
    ACTION_ARRAY,
    ACTION_KEY,
    BLUR_SCORE_KEY,
    DEFAULT_CAMERA,
    DEFAULT_REPO_ID,
    EPISODES_JSONL,
    FRAMES_JSONL,
    INFO_JSON,
    META_DIRNAME,
    SOURCE_FRAME_KEY,
    STATE_ARRAY,
    STATE_KEY,
    canonical_camera_order,
    probe_jpeg,
    read_jsonl,
    setup_logging,
)

@dataclass(frozen=True)
class CameraSpec:
    """一路相机的特征名与声明尺寸。"""

    name: str
    key: str
    height: int
    width: int


@dataclass
class EpisodeFrames:
    """一个 episode 及其每一路相机的帧清单（尚未落盘）。"""

    episode_index: int
    episode: str
    source_video: Optional[str]
    cameras: List[str]
    source_fps: Optional[float]
    frames_by_camera: Dict[str, List[dict]]
    length: int = 0
    has_episode_key: bool = True
    declared_cameras: Optional[List[str]] = None
    declared_length: Optional[int] = None

    @property
    def frames(self) -> List[dict]:
        """主相机帧，供兼容/报告使用。"""
        return self.frames_by_camera[self.cameras[0]]


def load_info(dataset_dir: Path) -> dict:
    path = dataset_dir / META_DIRNAME / INFO_JSON
    if not path.is_file():
        raise ValueError(f"缺少文件：{path}（请先跑 format_dataset.py）")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} 的内容不是 JSON 对象")
    return payload


def resolve_camera_shape(info: dict, camera_key: str) -> Tuple[int, int]:
    """兼容包装：从 info.json 取单个相机的 (height, width)。"""
    features = info.get("features")
    if not isinstance(features, dict) or camera_key not in features:
        raise ValueError(
            f"info.json 里没有相机特征 {camera_key}；"
            f"已有的特征是：{sorted((features or {}).keys())}"
        )
    shape = features[camera_key].get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"info.json 里 {camera_key} 的 shape 不合法：{shape!r}")
    height, width = shape[0], shape[1]
    if not (isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0):
        raise ValueError(f"info.json 里 {camera_key} 的宽高不合法：{shape!r}")
    return height, width


def resolve_cameras(info: dict, override: Optional[str] = None) -> List[CameraSpec]:
    """从 info.json 解析所有相机；--camera 仅允许单相机覆盖。"""
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json 缺少 features")
    keys = [
        key
        for key in features
        if isinstance(key, str) and key.startswith("observation.images.")
    ]
    if not keys:
        raise ValueError("info.json 里没有任何 observation.images.* 特征")

    if override:
        key = f"observation.images.{override}"
        if key not in keys:
            raise ValueError(
                f"info.json 里没有相机特征 {key}；现有相机：{sorted(keys)}"
            )
        if len(keys) > 1:
            raise ValueError(
                "info.json 声明了多个相机，不能再传 --camera；"
                "相机名来自清单的 extras.cameras"
            )
        selected = [key]
    else:
        extras = info.get("extras") if isinstance(info.get("extras"), dict) else {}
        declared = extras.get("cameras")
        if isinstance(declared, list) and declared:
            selected = [f"observation.images.{name}" for name in declared]
            missing = [key for key in selected if key not in keys]
            if missing:
                raise ValueError(f"extras.cameras 中的特征不存在：{missing}")
        else:
            selected = [
                f"observation.images.{name}"
                for name in canonical_camera_order(
                    key[len("observation.images.") :] for key in keys
                )
            ]

    cameras = []
    for key in selected:
        height, width = resolve_camera_shape(info, key)
        cameras.append(
            CameraSpec(
                name=key[len("observation.images.") :],
                key=key,
                height=height,
                width=width,
            )
        )
    return cameras


def build_features(
    cameras_or_key,
    height: Optional[int] = None,
    width: Optional[int] = None,
    action_dim: Optional[int] = None,
) -> dict:
    """组装 lerobot 的 features 描述。

    兼容旧调用：传单个 camera_key + height/width；新代码传 CameraSpec 列表。
    """
    if isinstance(cameras_or_key, str):
        key = cameras_or_key
        if height is None or width is None:
            raise ValueError("单个 camera_key 调用必须传 height 和 width")
        cameras = [
            CameraSpec(
                name=key[len("observation.images.") :]
                if key.startswith("observation.images.")
                else key,
                key=key,
                height=height,
                width=width,
            )
        ]
    else:
        cameras = list(cameras_or_key)

    features = {}
    for camera in cameras:
        features[camera.key] = {
            "dtype": "video",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    features[SOURCE_FRAME_KEY] = {"dtype": "int64", "shape": (1,), "names": None}
    features[BLUR_SCORE_KEY] = {
        "dtype": "float32",
        "shape": (len(cameras),),
        "names": [camera.name for camera in cameras] if len(cameras) > 1 else None,
    }
    if action_dim is not None:
        # 与官方 lerobot/droid_1.0.1 的口径一致：8 维 = 7 关节 + 1 夹爪。
        features[ACTION_KEY] = {"dtype": "float32", "shape": (action_dim,), "names": None}
        features[STATE_KEY] = {"dtype": "float32", "shape": (action_dim,), "names": None}
    return features


def group_episodes(
    frames: Sequence[dict],
    episodes: Sequence[dict],
    camera_names: Optional[Sequence[str]] = None,
) -> List[EpisodeFrames]:
    """按 episode_index 把帧清单分组，并挂上 episode 级信息。

    一行帧清单对应一个 (相机, 帧位置)。同一 episode 的各路相机必须逐位置对齐：
    frame_index / source_frame / timestamp 序列完全一致。旧清单没有 ``camera``
    字段时按单相机处理（默认 front）。
    """
    fallback_camera = (
        camera_names[0] if camera_names and len(camera_names) == 1 else DEFAULT_CAMERA
    )

    by_episode: Dict[int, EpisodeFrames] = {}
    for row in episodes:
        index = row.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"episodes.jsonl 里有非法 episode_index：{row!r}")
        source_video = row.get("source_video")
        episode_key = row.get("episode")
        has_episode_key = isinstance(episode_key, str) and bool(episode_key)
        if not has_episode_key:
            episode_key = str(source_video if source_video is not None else index)
        declared = row.get("cameras")
        if declared is not None and (
            not isinstance(declared, list)
            or not declared
            or not all(isinstance(name, str) and name for name in declared)
        ):
            raise ValueError(
                f"episodes.jsonl 里 episode {index} 的 cameras 不合法：{declared!r}"
            )
        by_episode[index] = EpisodeFrames(
            episode_index=index,
            episode=str(episode_key),
            source_video=None if source_video is None else str(source_video),
            cameras=[],
            source_fps=row.get("source_fps"),
            frames_by_camera={},
            length=0,
            has_episode_key=has_episode_key,
            declared_cameras=list(declared) if declared else None,
            declared_length=row.get("length"),
        )

    for row in frames:
        index = row.get("episode_index")
        if index not in by_episode:
            raise ValueError(
                f"frames.jsonl 里的 episode_index {index!r} 在 episodes.jsonl 中不存在"
            )
        camera = row.get("camera")
        if camera is None:
            camera = fallback_camera
        elif not isinstance(camera, str) or not camera:
            raise ValueError(f"frames.jsonl 里非法 camera 字段：{row!r}")
        by_episode[index].frames_by_camera.setdefault(camera, []).append(row)

    ordered = [by_episode[key] for key in sorted(by_episode)]
    for episode in ordered:
        if not episode.frames_by_camera:
            raise ValueError(f"episode {episode.episode_index} 一帧都没有")

        actual = set(episode.frames_by_camera)
        if episode.declared_cameras is not None:
            declared_set = set(episode.declared_cameras)
            if actual != declared_set:
                missing = sorted(declared_set - actual)
                extra = sorted(actual - declared_set)
                raise ValueError(
                    f"episode {episode.episode_index}（{episode.episode}）的相机与 "
                    f"episodes.jsonl 不一致：缺少 {missing}，多出 {extra}"
                )
        episode.cameras = canonical_camera_order(actual)

        sequences: Dict[str, List[tuple]] = {}
        for camera in episode.cameras:
            rows = episode.frames_by_camera[camera]
            # 帧必须按 episode 内位置有序，否则写进去的顺序会错乱
            rows.sort(key=lambda row: row.get("frame_index", 0))
            positions = [row.get("frame_index") for row in rows]
            if len(set(positions)) != len(positions):
                raise ValueError(
                    f"episode {episode.episode_index}（{episode.episode}）的相机 "
                    f"{camera} 有重复 frame_index"
                )
            sequences[camera] = [
                (
                    row.get("frame_index"),
                    row.get("source_frame"),
                    row.get("timestamp"),
                )
                for row in rows
            ]

        reference_camera = episode.cameras[0]
        reference = sequences[reference_camera]
        for camera in episode.cameras[1:]:
            other = sequences[camera]
            if len(other) != len(reference):
                raise ValueError(
                    f"episode {episode.episode_index}（{episode.episode}）的相机帧数不同步："
                    f"{reference_camera}={len(reference)}，{camera}={len(other)}"
                )
            for position, (left, right) in enumerate(zip(reference, other)):
                if left != right:
                    raise ValueError(
                        f"episode {episode.episode_index}（{episode.episode}）的相机不同步："
                        f"第 {position} 个位置 {reference_camera}={left}，{camera}={right}"
                    )
        episode.length = len(reference)

        if isinstance(episode.declared_length, int) and not isinstance(
            episode.declared_length, bool
        ):
            if episode.declared_length != episode.length:
                raise ValueError(
                    f"episode {episode.episode_index}（{episode.episode}）的 length="
                    f"{episode.declared_length}，与帧清单的 {episode.length} 不符"
                )
    return ordered


def resolve_image_root(
    dataset_dir: Path, info: dict, override: Optional[str] = None
) -> Path:
    """确定第一步图片目录。

    默认读 info.json 的 `extras.image_root`（相对 dataset/ 解析）；`--images-root`
    可覆盖，方便移动过目录或只保留了清单的情况。
    """
    if override:
        return Path(override).expanduser().resolve()
    extras = info.get("extras") if isinstance(info.get("extras"), dict) else {}
    raw = extras.get("image_root")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "info.json 缺少 extras.image_root；请重跑 format_dataset.py，"
            "或显式传 --images-root"
        )
    root = Path(raw).expanduser()
    return (dataset_dir / root).resolve() if not root.is_absolute() else root.resolve()


def frame_path(image_root: Path, row: dict) -> Path:
    """把 frames.jsonl 里的相对路径拼到第一步图片目录上。"""
    raw = row.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"frames.jsonl 里有非法的 path 字段：{row!r}")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"frames.jsonl 里的 path 必须是相对路径，实际是 {raw!r}")
    return image_root / relative


def load_actions(dataset_dir: Path) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """读 `dataset/meta/actions/*.npz`，返回 {源视频: (action, state)}。

    目录不存在就返回 None，表示这个数据集不带动作、一切照旧。文件名（去掉后缀）
    就是帧清单里的 `source_video`，所以取动作是一次字典查找，不做步长算术。
    """
    actions_dir = dataset_dir / META_DIRNAME / ACTIONS_DIRNAME
    if not actions_dir.is_dir():
        return None

    loaded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(actions_dir.glob("*.npz")):
        try:
            with np.load(path) as data:
                action = data[ACTION_ARRAY]
                state = data[STATE_ARRAY]
        except KeyError as exc:
            raise ValueError(
                f"{path} 里缺少数组 {exc}；请用 import_actions.py 重新生成"
            ) from exc
        except OSError as exc:
            raise ValueError(f"{path} 读不出来（文件损坏？）：{exc}") from exc
        if action.shape != state.shape:
            raise ValueError(
                f"{path} 的 action {action.shape} 与 state {state.shape} 形状不一致"
            )
        loaded[path.stem] = (action, state)
    return loaded or None


def action_dim_of(
    actions: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> Optional[int]:
    """所有源视频的动作维度必须一致（lerobot 只声明一份 shape）。返回那一维。"""
    if not actions:
        return None
    for name, (action, state) in actions.items():
        if action.ndim != 2:
            raise ValueError(f"{name} 的动作不是二维 (N, A)，实际是 {action.shape}")
    dims = {int(action.shape[1]) for action, _ in actions.values()}
    if len(dims) != 1:
        raise ValueError(f"各源视频的动作维度不一致：{sorted(dims)}")
    return dims.pop()


def action_key_for(
    episode: EpisodeFrames,
    actions: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[str]:
    """找到该 episode 的动作侧车键。

    新清单的权威键是 ``episode``；只有旧清单（没有 episode 字段）才回退到
    ``source_video``，避免多相机目录模式下静默配错动作。
    """
    candidates = [episode.episode]
    if not episode.has_episode_key and episode.source_video is not None:
        candidates.append(episode.source_video)
    for key in candidates:
        if key in actions:
            return key
    return None


def action_problem(
    episode: EpisodeFrames, row: dict, actions: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Optional[str]:
    """这一帧能不能取到动作？取不到就返回原因。

    动作一旦写进 features，lerobot 就要求每帧都提供 —— 也就意味着**每一帧都必须
    有对应动作**。缺的时候报错而不是填 0：静默填 0 等于告诉模型「此刻机器人什么
    也没做」，而且不会报错。
    """
    key = action_key_for(episode, actions)
    if key is None:
        candidates = [episode.episode]
        if not episode.has_episode_key and episode.source_video is not None:
            candidates.append(episode.source_video)
        return (
            f"episode {episode.episode_index}（{episode.episode}）没有动作文件"
            f"（查找键：{candidates}）。请对它跑 import_actions.py，"
            f"或把它从素材里排除：{row.get('path')}"
        )
    frame = row.get("source_frame")
    if isinstance(frame, bool) or not isinstance(frame, int):
        return f"帧清单里没有 source_frame，无法与动作对齐：{row.get('path')}"
    rows = actions[key][0].shape[0]
    if not 0 <= frame < rows:
        return (
            f"帧号 {frame} 超出动作范围（{key} 的动作只有 {rows} 行）："
            f"{row.get('path')}。动作文件与视频对不上？"
        )
    return None


def _cameras_for_check(
    episodes: Sequence[EpisodeFrames],
    height: int,
    width: int,
) -> List[CameraSpec]:
    """兼容旧调用的 (height, width)：按清单里的相机名建 spec。"""
    names = sorted({name for episode in episodes for name in episode.cameras})
    if not names:
        names = [DEFAULT_CAMERA]
    return [
        CameraSpec(name=name, key=f"observation.images.{name}", height=height, width=width)
        for name in names
    ]


def precheck_frames(
    episodes: Sequence[EpisodeFrames],
    image_root: Path,
    cameras_or_height,
    width: Optional[int] = None,
    actions: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[int, List[str]]:
    """落盘前先把所有帧验一遍，返回 (同步采样数, 问题列表)。

    必须在创建数据集之前做：lerobot 一旦开始写，中途报错会留下半成品目录。
    只读 JPEG 头部，两千多帧约一两秒。

    cameras_or_height 可传 CameraSpec 列表，也可传旧式 height（配合 width）。
    """
    if isinstance(cameras_or_height, int):
        if width is None:
            raise ValueError("旧式调用必须同时传 height 和 width")
        cameras = _cameras_for_check(episodes, cameras_or_height, width)
    else:
        cameras = list(cameras_or_height)

    expected = [camera.name for camera in cameras]
    problems: List[str] = []
    total = 0
    for episode in episodes:
        extra = [name for name in episode.cameras if name not in expected]
        missing = [name for name in expected if name not in episode.cameras]
        if extra or missing:
            problems.append(
                f"episode {episode.episode_index}（{episode.episode}）的相机与 info.json "
                f"不一致：缺少 {missing}，多出 {extra}"
            )
            if len(problems) >= 5:
                problems.append("……（更多问题已省略）")
                return total, problems
            continue

        reference = episode.frames
        for position, reference_row in enumerate(reference):
            total += 1
            if actions is not None:
                missing_action = action_problem(episode, reference_row, actions)
                if missing_action:
                    problems.append(missing_action)
                    if len(problems) >= 5:
                        problems.append("……（更多问题已省略）")
                        return total, problems
                    continue

            for camera in cameras:
                row = episode.frames_by_camera[camera.name][position]
                path = frame_path(image_root, row)
                # blur_score 是必须的：特征一旦声明，lerobot 要求每帧都提供。
                # 报错而不是填默认值 —— 静默补 0 会让所有帧看起来都模糊。
                score = row.get("blur_score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    problems.append(
                        f"帧清单里没有可用的 blur_score（{score!r}，相机 "
                        f"{camera.name}）。上游是旧版 format_dataset.py？"
                        f"请重跑前两步：{row.get('path')}"
                    )
                    continue
                if not path.is_file():
                    problems.append(f"帧文件不存在（相机 {camera.name}）：{path}")
                    continue
                size = probe_jpeg(path)
                if size is None:
                    problems.append(f"JPEG 无法解析（损坏？）：{path}")
                elif size != (camera.width, camera.height):
                    problems.append(
                        f"尺寸不符（相机 {camera.name} 期望 "
                        f"{camera.width}x{camera.height}，实际 {size[0]}x{size[1]}）：{path}"
                    )
                if len(problems) >= 5:
                    problems.append("……（更多问题已省略）")
                    return total, problems
    return total, problems


def check_encoder_drops(dataset, label: str, logger: logging.Logger) -> None:
    """确认流式编码没有丢帧，丢了就报错退出。

    编码队列满时 lerobot 会**丢帧**，且只打一条 warning（见 video_utils.feed_frame）。
    视频因此会比 parquet 少几帧，帧网格重新变得不均匀 —— 正是这个管线要消灭的
    失败模式，不能让它静默发生。

    读的是 lerobot 的私有计数。拿不到不算失败（免得它升级就崩），但会告警说明
    「这次没能核对」，而不是假装核对过了。
    """
    # 挂载位置随版本变过：0.4.x 是 dataset._streaming_encoder，0.6 起 create() 把它
    # 交给 DatasetWriter 保管，于是变成 dataset.writer._streaming_encoder。
    # 也就是说现在这一条依赖着两处私有结构，比升级前更脆。
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    dropped = getattr(encoder, "_dropped_frames", None)
    if not isinstance(dropped, dict):
        logger.warning(
            "%s：读不到 lerobot 的丢帧计数（_streaming_encoder._dropped_frames 变了？），"
            "本次无法核对视频帧数是否与清单一致",
            label,
        )
        return

    total = sum(int(count) for count in dropped.values())
    if total:
        raise RuntimeError(
            f"{label} 的流式编码丢了 {total} 帧：视频帧数少于清单，帧网格会不再均匀。"
            "请换更快的编码器（--vcodec h264 或 auto）、或关掉流式编码"
        )


def check_frame_grid(episodes: Sequence[EpisodeFrames], fps: int) -> List[str]:
    """检查帧是否落在 1/fps 的均匀网格上，返回告警列表（空列表 = 没问题）。

    lerobot 的 timestamp 恒等于 frame_index / fps（add_frame() 里写死），delta_timestamps
    也按帧索引偏移换算。只有帧确实等间隔排列时这套假设才成立。

    这是告警而不是错误：数据集照样写得出来、加载得了，只是时间戳与真实时间不符。
    """
    warnings: List[str] = []
    for episode in episodes:
        label = f"episode {episode.episode_index}（{episode.episode}）"
        frames = episode.frames
        # source_frame 在 write_dataset 里是可缺省的（退回帧位置），缺了就无从判断间隔。
        # 用默认值 0 硬凑会算出错误的间隔并误报，所以这种情况直接跳过间隔检查。
        gaps = (
            []
            if any("source_frame" not in row for row in frames)
            else [
                cur["source_frame"] - prev["source_frame"]
                for prev, cur in zip(frames, frames[1:])
            ]
        )

        if gaps and len(set(gaps)) > 1:
            warnings.append(
                f"{label} 的帧号间隔不均匀（共 {len(set(gaps))} 种，最大 {max(gaps)}）："
                f"帧不在均匀网格上，delta_timestamps 会取到错误的时间点。"
                f"这些帧是上游按 --blur-threshold 删除后留下的缺口？请重跑前两步"
            )

        source_fps = episode.source_fps
        if not isinstance(source_fps, (int, float)) or isinstance(source_fps, bool) or source_fps <= 0:
            warnings.append(
                f"{label} 没有可用的原始帧率，无法核对帧间隔是否等于 1/{fps} 秒；"
                f"lerobot 会按 frame_index / {fps} 写 timestamp"
            )
            continue

        step = source_fps / float(fps)
        rounded = round(step)
        # 容差按相对比例给：原始帧率常被容器记成 30.00005 这类带尾数的值
        if rounded < 1 or abs(step - rounded) > 1e-4 * rounded:
            drift_ms = abs(1.0 / fps - rounded / source_fps) * 1000
            warnings.append(
                f"{label} 的原始帧率 {source_fps:g} 不是处理帧率 {fps} 的整数倍"
                f"（商 {step:.6g}）：lerobot 按 frame_index / {fps} 写的 timestamp "
                f"每帧偏约 {drift_ms:.1f} ms，且会随帧数累积"
            )
        elif gaps and len(set(gaps)) == 1 and gaps[0] != rounded:
            # 帧是均匀的，但间隔与声明的帧率对不上 —— 记录与素材不是同一批
            warnings.append(
                f"{label} 的帧号间隔是 {gaps[0]}，而原始帧率 {source_fps:g} ÷ 处理帧率 "
                f"{fps} 应为 {rounded}：两者不符说明记录与素材对不上，"
                f"lerobot 的 timestamp 会偏离真实时间"
            )
    return warnings


def write_dataset(
    episodes: Sequence[EpisodeFrames],
    image_root: Path,
    output_dir: Path,
    repo_id: str,
    robot_type: str,
    cameras: Sequence[CameraSpec],
    fps: int,
    task: str,
    total_frames: int,
    vcodec: str,
    actions: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    logger: logging.Logger,
) -> int:
    """调 lerobot 的官方 API 把数据集写出来，返回写入的同步采样数。"""
    # 延迟导入：lerobot 会连带拉起 torch，要好几秒。放函数里，
    # 本模块不碰 lerobot 的纯逻辑测试就能保持秒级。
    from lerobot import __version__ as lerobot_version
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    # 版本决定格式（0.3.x 写 v2.1，0.4.x 起写 v3.0），而本项目针对 0.6.1 / v3.0 开发验证。
    # requirements.txt 里已 pin，但那只挡 pip 安装；这里把实际版本打出来便于排查。
    logger.info("lerobot %s，输出格式 %s", lerobot_version, CODEBASE_VERSION)
    if CODEBASE_VERSION != "v3.0":
        logger.warning(
            "当前 lerobot 输出的是 %s 格式，本项目针对 v3.0 开发验证，"
            "产物布局可能与 README 描述不一致",
            CODEBASE_VERSION,
        )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(
            cameras, action_dim=action_dim_of(actions)
        ),
        root=output_dir,
        use_videos=True,
        robot_type=robot_type,
        # 0.6 起编码参数收进 RGBEncoderConfig（0.5 及以前是 create() 的直接参数）。
        # 只传 vcodec，其余保持默认：g=2、crf=30、pix_fmt=yuv420p —— 与旧版一致，
        # 产物不变。将来要调 GOP/质量就改这里。
        rgb_encoder=RGBEncoderConfig(vcodec=vcodec),
        # 不开的话，帧会先落成 PNG 到临时目录、等 episode 结束再统一编码 ——
        # 等于把中间产物又写一遍。开着则直接流式喂给编码线程，save_episode()
        # 几乎是瞬时的，内存也不随 episode 长度增长。
        streaming_encoding=True,
        # 默认只缓冲 30 帧，编码稍慢就丢帧（丢帧的后果见 check_encoder_drops）。
        # 调大给突发留余量；真跟不上会被 check_encoder_drops 抓住并报错。
        encoder_queue_maxsize=300,
    )
    # 只报「请求的」编码器：0.6 起 dataset 上不再有 .vcodec 属性（也不去碰私有字段），
    # 而 --vcodec auto 实际选中哪个由 lerobot 自己打日志（"Auto-selected video codec"）
    logger.info("视频编码器：%s（流式编码已开启）", vcodec)

    written = 0
    non_rgb_warned = False
    try:
        with tqdm(total=total_frames, desc="写入帧", unit="frame") as bar:
            for episode in episodes:
                action_key = (
                    action_key_for(episode, actions) if actions is not None else None
                )
                if actions is not None and action_key is None:
                    raise RuntimeError(
                        f"episode {episode.episode_index}（{episode.episode}）"
                        "在 precheck 之后仍找不到动作侧车"
                    )

                for position, reference_row in enumerate(episode.frames):
                    frame_data = {
                        # lerobot 每帧必填；没有标注就传空串
                        "task": task,
                    }
                    blur_scores = []
                    for camera in cameras:
                        row = episode.frames_by_camera[camera.name][position]
                        source = frame_path(image_root, row)
                        image = Image.open(source)
                        image.load()  # 立刻读进内存，别让句柄留在文件上

                        # 声明的是 3 通道；灰度 / CMYK 的 JPEG 必须转成 RGB 才能喂编码器
                        if image.mode != "RGB":
                            if not non_rgb_warned:
                                logger.warning(
                                    "图片不是 RGB 模式（%s：%s），已自动转换；"
                                    "数据集声明的通道数是 3",
                                    source.name,
                                    image.mode,
                                )
                                non_rgb_warned = True
                            image = image.convert("RGB")

                        # 流式编码器要的是 numpy 数组（(H,W,C) 或 (C,H,W)），
                        # 喂 PIL 对象会在编码线程里炸 "no attribute 'shape'"
                        frame_data[camera.key] = np.asarray(image, dtype=np.uint8)
                        blur_scores.append(float(row["blur_score"]))

                    # 同一帧位置各相机共享 source_frame；blur 按相机顺序组成向量
                    source_frame = int(reference_row.get("source_frame", position))
                    frame_data[SOURCE_FRAME_KEY] = np.array(
                        [source_frame], dtype=np.int64
                    )
                    frame_data[BLUR_SCORE_KEY] = np.asarray(
                        blur_scores, dtype=np.float32
                    )
                    if actions is not None:
                        # 按**源视频帧号**取行，不做步长算术 —— 这也正是 source_frame
                        # 被保留成正式列的原因。范围与存在性已由 precheck 保证。
                        action, state = actions[action_key]
                        frame_data[ACTION_KEY] = np.asarray(action[source_frame], dtype=np.float32)
                        frame_data[STATE_KEY] = np.asarray(state[source_frame], dtype=np.float32)
                    dataset.add_frame(frame_data)
                    written += 1
                    bar.update(1)

                dataset.save_episode()
                # 必须在 save_episode 之后立刻查：丢帧计数在下一个 start_episode 时清零
                check_encoder_drops(
                    dataset,
                    f"episode {episode.episode_index}（{episode.episode}）",
                    logger,
                )
                logger.info(
                    "episode %d（%s）完成：%d 个采样 × %d 路相机",
                    episode.episode_index,
                    episode.episode,
                    episode.length,
                    len(cameras),
                )
    except Exception:
        logger.exception(
            "写入中断。%s 是半成品（parquet 没有 footer，无法加载）；"
            "修好问题后加 --force 重新运行即可重建",
            output_dir,
        )
        raise

    # 不调 finalize() 就不写 parquet footer，数据集无法加载
    dataset.finalize()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 format_dataset.py 的产物组装成 lerobot v3.0 格式的数据集。"
    )
    parser.add_argument(
        "--input", "-i", default="dataset", help="第二步的产物目录（默认 dataset）"
    )
    parser.add_argument(
        "--output", "-o", default="lerobot_dataset", help="数据集输出目录（默认 lerobot_dataset）"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"数据集标识（lerobot 必填，本质是 HF Hub 路径格式；默认 {DEFAULT_REPO_ID}）",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="只导出指定相机（仅单相机清单可用）；缺省导出 info.json 声明的全部相机",
    )
    parser.add_argument(
        "--images-root",
        default=None,
        help="第一步的图片目录；缺省时读 info.json 的 extras.image_root",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="处理帧率；缺省时沿用第二步 info.json 里的值"
    )
    parser.add_argument(
        "--task", default="", help="每帧的任务标注（默认空串，表示没有标注）"
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="视频编码器：h264/hevc/libsvtav1/auto，或硬件编码器名（默认 h264）。"
        "libsvtav1 压缩率更高但编码/解码更慢；"
        "auto 会选硬件编码器（编码更快，但对小尺寸帧会失败）",
    )
    parser.add_argument(
        "--force", action="store_true", help="输出目录已存在时整目录删掉重建"
    )

    parser.add_argument(
        "--robot-type", default="unknown", help="机器人类型（写入 lerobot info）"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("export_lerobot")

    dataset_dir = Path(args.input)
    output_dir = Path(args.output)

    if not dataset_dir.is_dir():
        logger.error("输入目录不存在或不是目录：%s", dataset_dir)
        return 1

    try:
        info = load_info(dataset_dir)
        frames = read_jsonl(
            dataset_dir / META_DIRNAME / FRAMES_JSONL, missing_hint="请先跑 format_dataset.py"
        )
        episode_rows = read_jsonl(
            dataset_dir / META_DIRNAME / EPISODES_JSONL, missing_hint="请先跑 format_dataset.py"
        )
        cameras = resolve_cameras(info, args.camera)
        image_root = resolve_image_root(dataset_dir, info, args.images_root)
        episodes = group_episodes(
            frames, episode_rows, camera_names=[camera.name for camera in cameras]
        )
        # 自动发现，不做成开关：「有没有动作」是事实而不是选择
        actions = load_actions(dataset_dir)
        action_dim = action_dim_of(actions)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if not image_root.is_dir():
        logger.error(
            "图片目录不存在：%s。请检查 info.json 的 extras.image_root，"
            "或用 --images-root 指定第一步的 output/images",
            image_root,
        )
        return 1

    if actions is None:
        logger.info(
            "未发现动作侧车（%s/meta/%s/），数据集不含 action/observation.state —— "
            "「能不能训练」只剩 observation，见 README「已知限制」",
            dataset_dir,
            ACTIONS_DIRNAME,
        )
    else:
        logger.info(
            "发现 %d 个 episode 的动作侧车（%d 维），数据集将带上 action/observation.state",
            len(actions),
            action_dim,
        )

    # 帧率：默认沿用第二步的记录，显式指定则以命令行优先
    info_fps = info.get("fps")
    if args.fps is None:
        if not isinstance(info_fps, int) or info_fps <= 0:
            logger.error("info.json 里没有可用的 fps，请显式指定 --fps")
            return 1
        fps = info_fps
    else:
        fps = args.fps
        if isinstance(info_fps, int) and info_fps != fps:
            logger.warning(
                "info.json 记录的帧率是 %d，与本次 --fps %d 不一致，请确认是否为同一批数据",
                info_fps,
                fps,
            )
    if fps <= 0:
        logger.error("--fps 必须为正整数")
        return 1

    if not str(args.repo_id).strip():
        logger.error("--repo-id 不能为空")
        return 1
    if not str(args.robot_type).strip():
        logger.error("--robot-type 不能为空")
        return 1

    # 校验必须在删旧目录之前：--force 会整个删掉已有数据集，若输入有问题就白丢了
    total_frames, problems = precheck_frames(
        episodes, image_root, cameras, actions=actions
    )
    if problems:
        logger.error("有 %d 个帧文件不可用，未开始写入：", len(problems))
        for problem in problems:
            logger.error("  %s", problem)
        return 1

    # 时间轴告警不中断：数据集本身仍然有效，只是 timestamp 不代表真实时间
    for grid_warning in check_frame_grid(episodes, fps):
        logger.warning("%s", grid_warning)

    # lerobot 的 create() 要求输出目录不存在，所以 --force 就是整目录重建。
    # 重建结果必然是干净的，不会有上一轮的残渣。
    if output_dir.exists():
        if not args.force:
            logger.error("输出目录已存在：%s。如需重建请加 --force", output_dir)
            return 1
        logger.info("--force：删除已存在的 %s", output_dir)
        try:
            shutil.rmtree(output_dir)
        except OSError:
            logger.exception("删除失败：%s", output_dir)
            return 1

    logger.info(
        "准备写入：%d 个 episode，%d 个同步采样，%d 路相机，fps=%d，图片目录=%s",
        len(episodes),
        total_frames,
        len(cameras),
        fps,
        image_root,
    )
    for camera in cameras:
        logger.info(
            "  相机 %s：%dx%d（%s）",
            camera.name,
            camera.width,
            camera.height,
            camera.key,
        )
    if not args.task:
        logger.info("--task 为空，meta/tasks.parquet 里会是一行空任务（表示没有标注）")

    start_ts = time.time()
    try:
        written = write_dataset(
            episodes=episodes,
            image_root=image_root,
            output_dir=output_dir,
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            cameras=cameras,
            fps=fps,
            task=args.task,
            total_frames=total_frames,
            vcodec=args.vcodec,
            actions=actions,
            logger=logger,
        )
    except Exception:
        return 1

    elapsed = time.time() - start_ts
    logger.info("=== 完成 ===")
    logger.info(
        "episodes=%d frames=%d 耗时 %.1fs（%.0f ms/帧）",
        len(episodes),
        written,
        elapsed,
        elapsed / written * 1000 if written else 0.0,
    )
    logger.info("输出目录：%s", output_dir.resolve())
    logger.info(
        "加载方式：LeRobotDataset(%r, root=%r)",
        args.repo_id,
        str(output_dir.resolve()),
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
