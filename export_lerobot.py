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

import format_dataset  # 复用 JPEG 头部解析：纯标准库，import 很轻

# format_dataset.py 写在 dataset/meta/ 下的文件名
FRAMES_JSONL = "frames.jsonl"
EPISODES_JSONL = "episodes.jsonl"
INFO_JSON = "info.json"

DEFAULT_REPO_ID = "local/embodied-videos"
DEFAULT_CAMERA = "front"

# 自定义特征：原始视频里的帧号，将来 join 动作数据要靠它。
# shape 声明成 (1,)：lerobot 会映射成标量列，喂的时候要传长度 1 的数组。
SOURCE_FRAME_KEY = "source_frame"
# 自定义特征：该帧的 Laplacian 方差，由 clean_video.py 算出。
# 只写原始分数，不写「是否模糊」的布尔量 —— 阈值由下游自己定。
BLUR_SCORE_KEY = "blur_score"


@dataclass
class EpisodeFrames:
    """一个 episode 及其帧清单（尚未落盘）。"""

    episode_index: int
    source_video: str
    source_fps: Optional[float]
    frames: List[dict]


def load_jsonl(path: Path) -> List[dict]:
    """读一个 jsonl 文件；不存在或内容非法时抛 ValueError。"""
    if not path.is_file():
        raise ValueError(f"缺少文件：{path}（请先跑 format_dataset.py）")
    rows: List[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def load_info(dataset_dir: Path) -> dict:
    path = dataset_dir / "meta" / INFO_JSON
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
    """从 info.json 取该相机的 (height, width)。

    沿用 format_dataset.py 的众数决定 —— lerobot 要求每帧尺寸都与声明一致。
    """
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


def build_features(camera_key: str, height: int, width: int) -> dict:
    """组装 lerobot 的 features 描述。

    这里是唯一描述数据集有哪些特征的地方；将来接动作数据只改这个函数。

    相机标 "video" 而不是 "image"：前者把帧编码成 mp4，后者把图片字节直接嵌进
    parquet。图片模式有两个在大数据量下会爆的限制 —— 单个 arrow 批次超过 2 GB
    （32 位偏移）会报错，以及一个 episode 的全部帧要同时驻留内存。视频模式两个
    都没有，代价是有损编码。
    """
    return {
        camera_key: {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        SOURCE_FRAME_KEY: {"dtype": "int64", "shape": (1,), "names": None},
        BLUR_SCORE_KEY: {"dtype": "float32", "shape": (1,), "names": None},
    }


def group_episodes(
    frames: Sequence[dict], episodes: Sequence[dict]
) -> List[EpisodeFrames]:
    """按 episode_index 把帧清单分组，并挂上 episode 级信息。"""
    by_episode: Dict[int, EpisodeFrames] = {}
    for row in episodes:
        index = row.get("episode_index")
        if not isinstance(index, int):
            raise ValueError(f"episodes.jsonl 里有非法 episode_index：{row!r}")
        by_episode[index] = EpisodeFrames(
            episode_index=index,
            source_video=str(row.get("source_video", index)),
            source_fps=row.get("source_fps"),
            frames=[],
        )

    for row in frames:
        index = row.get("episode_index")
        if index not in by_episode:
            raise ValueError(
                f"frames.jsonl 里的 episode_index {index!r} 在 episodes.jsonl 中不存在"
            )
        by_episode[index].frames.append(row)

    ordered = [by_episode[key] for key in sorted(by_episode)]
    for episode in ordered:
        if not episode.frames:
            raise ValueError(f"episode {episode.episode_index} 一帧都没有")
        # 帧必须按 episode 内位置有序，否则写进去的顺序会错乱
        episode.frames.sort(key=lambda row: row.get("frame_index", 0))
    return ordered


def frame_path(dataset_dir: Path, row: dict) -> Path:
    """把 frames.jsonl 里的相对路径还原成绝对路径（拼上 dataset/ 根目录）。"""
    raw = row.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"frames.jsonl 里有非法的 path 字段：{row!r}")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"frames.jsonl 里的 path 必须是相对路径，实际是 {raw!r}")
    return dataset_dir / relative


def precheck_frames(
    episodes: Sequence[EpisodeFrames],
    dataset_dir: Path,
    height: int,
    width: int,
) -> Tuple[int, List[str]]:
    """落盘前先把所有帧验一遍，返回 (帧总数, 问题列表)。

    必须在创建数据集之前做：lerobot 一旦开始写，中途报错会留下半成品目录。
    只读 JPEG 头部，两千多帧约一两秒。
    """
    problems: List[str] = []
    total = 0
    for episode in episodes:
        for row in episode.frames:
            path = frame_path(dataset_dir, row)
            total += 1
            # blur_score 是必须的：特征一旦声明，lerobot 要求每帧都提供。
            # 报错而不是填默认值 —— 静默补 0 会让所有帧看起来都模糊。
            score = row.get("blur_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                problems.append(
                    f"帧清单里没有可用的 blur_score（{score!r}）。"
                    f"上游是旧版 format_dataset.py？请重跑前两步：{row.get('path')}"
                )
                continue
            if not path.is_file():
                problems.append(f"帧文件不存在：{path}")
                continue
            size = format_dataset.probe_jpeg(path)
            if size is None:
                problems.append(f"JPEG 无法解析（损坏？）：{path}")
            elif size != (width, height):
                problems.append(
                    f"尺寸不符（期望 {width}x{height}，实际 {size[0]}x{size[1]}）：{path}"
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
    encoder = getattr(dataset, "_streaming_encoder", None)
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
        label = f"episode {episode.episode_index}（{episode.source_video}）"
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
    dataset_dir: Path,
    output_dir: Path,
    repo_id: str,
    camera_key: str,
    height: int,
    width: int,
    fps: int,
    task: str,
    total_frames: int,
    vcodec: str,
    logger: logging.Logger,
) -> int:
    """调 lerobot 的官方 API 把数据集写出来，返回写入的帧数。"""
    # 延迟导入：lerobot 会连带拉起 torch，要好几秒。放函数里，
    # 本模块不碰 lerobot 的纯逻辑测试就能保持秒级。
    from lerobot import __version__ as lerobot_version
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    # 版本决定格式（0.3.x 写 v2.1，0.4.x 写 v3.0），而本项目针对 0.4.4 / v3.0 开发验证。
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
        features=build_features(camera_key, height, width),
        root=output_dir,
        use_videos=True,
        robot_type="unknown",
        vcodec=vcodec,
        # 不开的话，帧会先落成 PNG 到临时目录、等 episode 结束再统一编码 ——
        # 等于把中间产物又写一遍。开着则直接流式喂给编码线程，save_episode()
        # 几乎是瞬时的，内存也不随 episode 长度增长。
        streaming_encoding=True,
        # 默认只缓冲 30 帧，编码稍慢就丢帧（丢帧的后果见 check_encoder_drops）。
        # 调大给突发留余量；真跟不上会被 check_encoder_drops 抓住并报错。
        encoder_queue_maxsize=300,
    )
    logger.info("视频编码器：%s（流式编码已开启）", dataset.vcodec)

    written = 0
    non_rgb_warned = False
    try:
        with tqdm(total=total_frames, desc="写入帧", unit="frame") as bar:
            for episode in episodes:
                for position, row in enumerate(episode.frames):
                    source = frame_path(dataset_dir, row)
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

                    dataset.add_frame(
                        {
                            # 流式编码器要的是 numpy 数组（(H,W,C) 或 (C,H,W)），
                            # 喂 PIL 对象会在编码线程里炸 "no attribute 'shape'"
                            camera_key: np.asarray(image, dtype=np.uint8),
                            SOURCE_FRAME_KEY: np.array(
                                [int(row.get("source_frame", position))], dtype=np.int64
                            ),
                            BLUR_SCORE_KEY: np.array(
                                [float(row["blur_score"])], dtype=np.float32
                            ),
                            # lerobot 每帧必填；没有标注就传空串
                            "task": task,
                        }
                    )
                    written += 1
                    bar.update(1)

                dataset.save_episode()
                # 必须在 save_episode 之后立刻查：丢帧计数在下一个 start_episode 时清零
                check_encoder_drops(
                    dataset, f"episode {episode.episode_index}（{episode.source_video}）", logger
                )
                logger.info(
                    "episode %d（%s）完成：%d 帧",
                    episode.episode_index,
                    episode.source_video,
                    len(episode.frames),
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
        "--camera", default=DEFAULT_CAMERA, help=f"相机名，对应 observation.images.<camera>（默认 {DEFAULT_CAMERA}）"
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
        "libsvtav1 解码约为图片的 5 倍慢、h264 约 2.2 倍，训练取数时差别明显；"
        "auto 会选硬件编码器（编码更快，但对小尺寸帧会失败）",
    )
    parser.add_argument(
        "--force", action="store_true", help="输出目录已存在时整目录删掉重建"
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("export_lerobot")

    dataset_dir = Path(args.input)
    output_dir = Path(args.output)
    camera_key = f"observation.images.{args.camera}"

    if not dataset_dir.is_dir():
        logger.error("输入目录不存在或不是目录：%s", dataset_dir)
        return 1

    try:
        info = load_info(dataset_dir)
        frames = load_jsonl(dataset_dir / "meta" / FRAMES_JSONL)
        episode_rows = load_jsonl(dataset_dir / "meta" / EPISODES_JSONL)
        height, width = resolve_camera_shape(info, camera_key)
        episodes = group_episodes(frames, episode_rows)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

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

    # 校验必须在删旧目录之前：--force 会整个删掉已有数据集，若输入有问题就白丢了
    total_frames, problems = precheck_frames(episodes, dataset_dir, height, width)
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
        "准备写入：%d 个 episode，%d 帧，%dx%d，fps=%d，相机=%s",
        len(episodes),
        total_frames,
        width,
        height,
        fps,
        camera_key,
    )
    if not args.task:
        logger.info("--task 为空，meta/tasks.parquet 里会是一行空任务（表示没有标注）")

    start_ts = time.time()
    try:
        written = write_dataset(
            episodes=episodes,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            repo_id=args.repo_id,
            camera_key=camera_key,
            height=height,
            width=width,
            fps=fps,
            task=args.task,
            total_frames=total_frames,
            vcodec=args.vcodec,
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
