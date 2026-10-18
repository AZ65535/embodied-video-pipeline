import argparse
import concurrent.futures
import json
import logging
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional, Sequence, Tuple, List
import time

import cv2
from tqdm import tqdm

from pipeline_common import (
    BLUR_SCORES_FILENAME,
    DEFAULT_CAMERA,
    IMAGES_META_FILENAME,
    atomic_write_json,
    canonical_camera_order,
    episode_key_for_dir,
    frame_filename,
    is_blurry,
    load_camera_names,
    natural_key,
    setup_logging,
    validate_camera_name,
    video_output_prefix,
)


def list_videos(input_dir: Path) -> Iterable[Path]:
    """返回输入目录下的所有 mp4 视频路径（递归，后缀大小写不敏感）。

    单次遍历后过滤：Windows 上 glob 大小写不敏感，两轮 rglob 会重复匹配同一文件。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            yield p


def assign_output_names(videos: Iterable[Path], input_dir: Path) -> Dict[Path, str]:
    """给每个视频分配输出前缀；大小写不敏感重名时抛 ValueError。"""
    assigned: Dict[Path, str] = {}
    owner: Dict[str, Path] = {}
    for video in videos:
        prefix = video_output_prefix(video, input_dir)
        key = prefix.casefold()
        if key in owner:
            raise ValueError(
                f"输出名冲突：{owner[key]} 与 {video} 都会写成 {prefix}_frame_*.jpg，"
                "并发写入会互相覆盖；请重命名其中一个，或调整目录结构"
            )
        owner[key] = video
        assigned[video] = prefix
    return assigned


@dataclass(frozen=True)
class VideoPlan:
    """一个 mp4 在分组后的身份。"""

    path: Path
    output_prefix: str
    episode: str
    episode_dir: str
    camera: str
    source_stem: str


def build_video_plans(
    videos: Sequence[Path],
    input_dir: Path,
    group_by: str,
    camera: Optional[str],
    camera_names: Dict[str, str],
) -> List[VideoPlan]:
    """在解码前完成分组、命名和冲突校验。"""
    output_names = assign_output_names(videos, input_dir)
    plans: List[VideoPlan] = []
    episode_cameras: Dict[str, List[str]] = {}

    for video in sorted(videos, key=lambda path: natural_key(str(path))):
        prefix = output_names[video]
        stem = video.stem
        if group_by == "dir":
            relative_parent = video.parent.relative_to(input_dir)
            episode = episode_key_for_dir(relative_parent)
            camera_name = camera_names.get(stem, stem)
            episode_dir = relative_parent.as_posix() or "."
        else:
            episode = prefix
            camera_name = camera_names.get(stem, camera or DEFAULT_CAMERA)
            episode_dir = video.parent.relative_to(input_dir).as_posix() or "."
        reason = validate_camera_name(camera_name)
        if reason:
            raise ValueError(f"{video} 的相机名 {camera_name!r} 非法：{reason}")
        plans.append(
            VideoPlan(
                path=video,
                output_prefix=prefix,
                episode=episode,
                episode_dir=episode_dir,
                camera=camera_name,
                source_stem=stem,
            )
        )
        episode_cameras.setdefault(episode, []).append(camera_name)

    episodes_seen: Dict[str, str] = {}
    for plan in plans:
        key = plan.episode.casefold()
        if key in episodes_seen and episodes_seen[key] != plan.episode:
            raise ValueError(
                f"episode key 冲突：{episodes_seen[key]} 与 {plan.episode}（大小写不敏感）"
            )
        episodes_seen[key] = plan.episode
        cameras = episode_cameras[plan.episode]
        folded = [name.casefold() for name in cameras]
        if len(set(folded)) != len(folded):
            raise ValueError(f"{plan.episode} 的相机名有重复：{cameras}")

    camera_sets = {frozenset(names) for names in episode_cameras.values()}
    if len(camera_sets) > 1:
        raise ValueError(
            "不同 episode 的相机集合不一致："
            + "; ".join(
                f"{episode}: {sorted(names)}"
                for episode, names in sorted(episode_cameras.items())
            )
        )
    return plans


def blur_score(frame) -> float:
    """Laplacian 方差，越大越清晰。

    calibrate_blur.py 也调用这个函数算分位数 —— 必须共用同一个实现，
    否则标定出的分布与实际过滤时看到的分数不是一回事，标定就没有意义了。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_frame_blurry(frame, threshold: float) -> bool:
    # 阈值口径由 pipeline_common.is_blurry 单点定义；这里只负责先从画面上算分。
    return is_blurry(blur_score(frame), threshold)


def median_score(scores: Sequence[Tuple[int, float]]) -> float:
    """一组 (帧号, 分数) 的分数中位数；空集返回 0.0。

    中位数不需要阈值就能看出素材的模糊程度 —— calibrate_blur.py 的实测数据里
    正是靠它发现各视频相差 3.8 倍。
    """
    if not scores:
        return 0.0
    return float(statistics.median(score for _, score in scores))


def save_frame(frame, out_dir: Path, video_name: str, frame_number: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / frame_filename(video_name, frame_number)
    cv2.imwrite(str(out_path), frame)
    return out_path


class VideoResult(NamedTuple):
    """单个视频的处理结果。

    字段较多且类型相近，用具名元组而不是裸元组，免得读代码时数位置。
    """

    sampled: int
    elapsed: float
    video_name: str
    orig_fps: float
    scores: List[Tuple[int, float]]  # (原始帧号, 模糊分数)，按帧号升序


def process_video(
    video_path: Path,
    out_images_dir: Path,
    target_fps: int = 5,
    output_name: Optional[str] = None,
) -> VideoResult:
    """抽帧、算分，并保存全部采样帧。

    这里不丢弃任何帧：帧必须落在 1/fps 的均匀网格上，lerobot 的时序查询才成立。
    模糊与否只作为分数记录下来，判断留给 format_dataset.py 与下游。

    失败时 sampled 为 0、orig_fps 为 0.0。output_name 由 assign_output_names 统一分配。
    """
    logger = logging.getLogger(__name__)
    # 提前定义：异常路径下的 except 块需要返回它们
    video_name = output_name or video_path.stem
    sampled = 0
    orig_fps = 0.0
    scores: List[Tuple[int, float]] = []
    start_ts = time.time()

    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if orig_fps <= 0 or target_fps <= 0:
            frame_step = 1
        else:
            frame_step = max(1, int(round(orig_fps / float(target_fps))))

        # 进度条按「预期采样帧数」计，与实际调用 update 的次数对齐
        expected_samples = math.ceil(total_frames / frame_step) if total_frames > 0 else None

        read_idx = 0
        pbar = tqdm(total=expected_samples, desc=video_name, unit="frame")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if read_idx % frame_step == 0:
                    sampled += 1
                    try:
                        # 先算分再写盘：算分失败时不留下「有图无分」的帧。
                        # 反过来（写盘失败）会留下缺失文件，第三步的校验能抓到。
                        score = blur_score(frame)
                        save_frame(frame, out_images_dir, video_name, read_idx)
                        scores.append((read_idx, score))
                    except Exception:
                        logger.exception("Error processing frame %s of %s", read_idx, video_path)
                    pbar.update(1)
                read_idx += 1
        finally:
            # 实际可解码帧数可能与元数据不一致，收尾时补齐到 100%
            if pbar.total is not None and pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)
            pbar.close()

        elapsed = time.time() - start_ts
        logger.info(
            "Finished %s: sampled=%d median_blur_score=%.1f elapsed=%.2fs",
            video_path.name,
            sampled,
            median_score(scores),
            elapsed,
        )

        return VideoResult(sampled, elapsed, video_name, orig_fps, scores)

    except Exception:
        logger.exception("Failed to process video: %s", video_path)
        elapsed = time.time() - start_ts
        return VideoResult(sampled, elapsed, video_name, orig_fps, scores)

    finally:
        if cap is not None:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from videos and record each frame's blur score."
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory containing MP4 videos")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output base directory (images will be saved to OUTPUT/images/)",
    )
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second to sample")
    parser.add_argument(
        "--group-by",
        choices=["file", "dir"],
        default="file",
        help="file = 一个 mp4 一个 episode；dir = 同目录 mp4 是同一集的多路相机（默认 file）",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="file 模式的相机名（默认 front）；dir 模式请用 --camera-names",
    )
    parser.add_argument(
        "--camera-names",
        default=None,
        help="可选 JSON：文件名 stem -> 相机名，例如 {\"18026681\": \"wrist\"}",
    )
    return parser.parse_args()


def write_images_meta(
    out_images_dir: Path,
    fps_by_video: Dict[str, float],
    target_fps: int,
    results: Sequence[VideoResult],
    plans: Optional[Sequence[VideoPlan]] = None,
    group_by: str = "file",
    camera_names: Optional[Dict[str, str]] = None,
    source_root: Optional[Path] = None,
) -> None:
    """写 format 3 的 _meta.json：video 条目 + episode 分组。

    format 3 是第一步唯一支持的格式；旧格式会被忽略。相同 group_by 的重复运行
    按当前 episode 整集替换，未涉及的 episode 保留。

    `plans` 省略时按 file 模式从 results 合成，便于旧调用方/测试直接使用。
    """
    logger = logging.getLogger("clean_video")
    camera_names = camera_names or {}
    if plans is None:
        plans = [
            VideoPlan(
                path=Path(result.video_name),
                output_prefix=result.video_name,
                episode=result.video_name,
                episode_dir=".",
                camera=camera_names.get(result.video_name, DEFAULT_CAMERA),
                source_stem=result.video_name,
            )
            for result in results
        ]
    meta_path = out_images_dir / IMAGES_META_FILENAME

    payload: dict = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("已有的 %s 无法解析，将重新生成", meta_path)
            loaded = None
        if isinstance(loaded, dict) and loaded.get("format") == 3:
            if loaded.get("group_by") != group_by:
                raise ValueError(
                    f"{meta_path} 的 group_by={loaded.get('group_by')!r}，"
                    f"与本次 {group_by!r} 不一致；请换一个新的 --output"
                )
            payload = loaded
        elif loaded is not None:
            logger.warning("已有的 %s 不是 format 3，将被新格式替换", meta_path)

    videos = payload.get("videos") if isinstance(payload.get("videos"), dict) else {}
    episodes = payload.get("episodes") if isinstance(payload.get("episodes"), dict) else {}
    merged_names = payload.get("camera_names") if isinstance(payload.get("camera_names"), dict) else {}
    merged_names.update(camera_names)

    stats = {r.video_name: r for r in results}
    for plan in plans:
        result = stats.get(plan.output_prefix)
        recorded_fps = fps_by_video.get(
            plan.output_prefix, result.orig_fps if result is not None else 0.0
        )
        if result is None or recorded_fps <= 0:
            videos.pop(plan.output_prefix, None)
            continue
        videos[plan.output_prefix] = {
            "orig_fps": round(float(recorded_fps), 6),
            "frame_step": max(1, int(round(recorded_fps / float(target_fps))))
            if target_fps > 0
            else 1,
            "sampled": result.sampled,
            "score_median": round(median_score(result.scores), 4),
            "elapsed": round(result.elapsed, 3),
            "source_stem": plan.source_stem,
            "episode": plan.episode,
            "camera": plan.camera,
        }

    touched = {plan.episode for plan in plans}
    for episode in touched:
        plans_for_episode = [plan for plan in plans if plan.episode == episode]
        episodes[episode] = {
            "episode_dir": plans_for_episode[0].episode_dir,
            "cameras": {plan.camera: plan.output_prefix for plan in plans_for_episode},
        }

    camera_sets = {
        frozenset(info.get("cameras", {}))
        for info in episodes.values()
        if isinstance(info, dict)
    }
    if len(camera_sets) > 1:
        raise ValueError(
            "合并后的 _meta.json 中 episode 相机集合不一致："
            + "; ".join(
                f"{episode}: {sorted(info.get('cameras', {}))}"
                for episode, info in sorted(episodes.items())
                if isinstance(info, dict)
            )
        )
    cameras = canonical_camera_order(next(iter(camera_sets))) if camera_sets else []

    payload.update(
        {
            "format": 3,
            "tool": "clean_video.py",
            "target_fps": target_fps,
            "group_by": group_by,
            "source_root": (
                str(source_root.resolve()) if source_root else payload.get("source_root")
            ),
            "cameras": cameras,
            "camera_names": merged_names,
            "episodes": episodes,
            "videos": videos,
        }
    )

    if atomic_write_json(meta_path, payload, logger):
        logger.info(
            "已写入 %s（%d 个 episode，%d 个视频，%d 路相机）",
            meta_path,
            len(episodes),
            len(videos),
            len(cameras),
        )


def write_blur_scores(out_images_dir: Path, results: Sequence[VideoResult]) -> None:
    """把每个采样帧的模糊分数写到 out_images_dir/_scores.jsonl。

    和 _meta.json 一样由父进程统一写、先写临时文件再替换、重跑时按视频 merge
    （重跑某几个视频只替换它们的行，其余保留）。

    分数存原始值而不是「是否模糊」的布尔量：阈值还没标定，而且判断阈值归
    format_dataset.py 管，这里只负责记录事实。
    """
    logger = logging.getLogger("clean_video")
    scores_path = out_images_dir / BLUR_SCORES_FILENAME

    # video -> {source_frame: score}，先读旧的再按本次结果覆盖
    by_video: Dict[str, Dict[int, float]] = {}
    if scores_path.exists():
        try:
            for lineno, line in enumerate(
                scores_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    by_video.setdefault(str(row["video"]), {})[int(row["source_frame"])] = float(
                        row["blur_score"]
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    logger.warning("已有的 %s 第 %d 行无法解析，已跳过", scores_path, lineno)
        except OSError:
            logger.warning("已有的 %s 读取失败，将重新生成", scores_path)

    for result in results:
        if result.orig_fps <= 0:
            continue  # 失败的视频不写
        by_video[result.video_name] = {frame: score for frame, score in result.scores}

    # 排序只为让输出稳定可比对，与语义无关
    lines = [
        json.dumps(
            {"video": name, "source_frame": frame, "blur_score": round(score, 4)},
            ensure_ascii=False,
        )
        for name in sorted(by_video)
        for frame, score in sorted(by_video[name].items())
    ]

    tmp_path = scores_path.parent / (scores_path.name + ".tmp")
    try:
        tmp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp_path, scores_path)
    except OSError:
        logger.exception("写入 %s 失败", scores_path)
        return

    logger.info("已写入 %s（累计 %d 帧的模糊分数）", scores_path, len(lines))


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("clean_video")

    input_dir = Path(args.input)
    output_base = Path(args.output)
    out_images_dir = output_base / "images"

    try:
        out_images_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Cannot create output directory: %s", out_images_dir)
        return 1

    try:
        videos = list(list_videos(input_dir))
    except Exception:
        logger.exception("Failed to list videos in %s", input_dir)
        return 1

    if not videos:
        logger.info("No MP4 videos found in %s", input_dir)
        return 0

    if args.group_by == "dir" and args.camera is not None:
        logger.error("--group-by dir 不能同时传 --camera；相机名请用 --camera-names 或文件名")
        return 1

    try:
        camera_names = load_camera_names(
            Path(args.camera_names) if args.camera_names else None
        )
        plans = build_video_plans(
            videos, input_dir, args.group_by, args.camera, camera_names
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    nested = [plan for plan in plans if plan.output_prefix != plan.path.stem]
    if nested:
        logger.info(
            "有 %d 个视频位于子目录，输出名前缀带上了目录：%s",
            len(nested),
            ", ".join(
                f"{plan.path.name} -> {plan.output_prefix}" for plan in nested[:5]
            ),
        )

    total_sampled = 0
    results: List[VideoResult] = []
    results_by_prefix: Dict[str, VideoResult] = {}
    fps_by_video: Dict[str, float] = {}

    start_all = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(
                process_video_worker,
                str(plan.path),
                str(out_images_dir),
                args.fps,
                plan.output_prefix,
            ): plan
            for plan in plans
        }

        for fut in concurrent.futures.as_completed(future_map):
            plan = future_map[fut]
            try:
                result = fut.result()
            except Exception:
                logger.exception("Processing failed for %s", plan.path)
                continue

            results_by_prefix[plan.output_prefix] = result
            if result.orig_fps > 0:
                fps_by_video[result.video_name] = result.orig_fps
            results.append(result)
            total_sampled += result.sampled
            logger.info(
                "Completed %s: sampled=%d median_blur_score=%.1f elapsed=%.2fs",
                result.video_name,
                result.sampled,
                median_score(result.scores),
                result.elapsed,
            )

    total_elapsed = time.time() - start_all

    bad_episodes = set()
    if args.group_by == "dir":
        for plan in plans:
            result = results_by_prefix.get(plan.output_prefix)
            if result is None or result.orig_fps <= 0:
                bad_episodes.add(plan.episode)
                logger.error(
                    "episode %s 的相机 %s 处理失败，整集不可用",
                    plan.episode,
                    plan.camera,
                )
    plans_for_meta = [plan for plan in plans if plan.episode not in bad_episodes]

    # 报中位数而不是「过滤率」：中位数不需要阈值就能看出素材有多模糊
    logger.info("\n=== Benchmark Results ===")
    logger.info("%-40s %8s %10s %10s", "video", "sampled", "median", "time(s)")
    for result in results:
        logger.info(
            "%-40s %8d %10.1f %10.2f",
            result.video_name,
            result.sampled,
            median_score(result.scores),
            result.elapsed,
        )

    logger.info("Totals: sampled=%d total_time=%.2fs", total_sampled, total_elapsed)

    try:
        write_images_meta(
            out_images_dir,
            fps_by_video,
            args.fps,
            results,
            plans_for_meta,
            args.group_by,
            camera_names,
            input_dir,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    write_blur_scores(out_images_dir, results)
    return 1 if bad_episodes else 0


def process_video_worker(
    video_path_str: str,
    out_images_dir_str: str,
    target_fps: int,
    output_name: str,
):
    """可被进程池安全调用的包装函数（top-level）。"""
    video_path = Path(video_path_str)
    out_images_dir = Path(out_images_dir_str)
    return process_video(
        video_path,
        out_images_dir,
        target_fps=target_fps,
        output_name=output_name,
    )


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
