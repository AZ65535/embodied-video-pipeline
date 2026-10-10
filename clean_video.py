import argparse
import concurrent.futures
import json
import logging
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional, Sequence, Tuple, List
import time

import cv2
from tqdm import tqdm


# 处理参数与原始帧率的落盘文件名，写在图片目录里，供 format_dataset.py 读取。
IMAGES_META_FILENAME = "_meta.json"
# 每帧模糊分数的落盘文件名，同样供 format_dataset.py 读取。
BLUR_SCORES_FILENAME = "_scores.jsonl"


def list_videos(input_dir: Path) -> Iterable[Path]:
    """返回输入目录下的所有 mp4 视频路径（递归，后缀大小写不敏感）。

    单次遍历后过滤：Windows 上 glob 大小写不敏感，两轮 rglob 会重复匹配同一文件。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            yield p


def video_output_prefix(video_path: Path, input_dir: Path) -> str:
    """算出该视频的输出名前缀（图片文件名和 _meta.json 的键都用它）。

    规则：相对 input_dir 的路径去掉后缀，路径分隔符换成下划线。平铺目录下
    结果就是原来的 stem；子目录下带上目录名，避免同名视频互相覆盖。
    """
    try:
        relative = video_path.relative_to(input_dir)
    except ValueError:
        return video_path.stem  # 不在输入目录下（理论上不会发生）
    return "_".join(relative.with_suffix("").parts)


def assign_output_names(videos: Iterable[Path], input_dir: Path) -> Dict[Path, str]:
    """给每个视频分配输出前缀；仍有重名时抛 ValueError。

    带目录前缀后仍可能撞车（videos/a_1.mp4 与 videos/a/1.mp4 都得到 "a_1"）。
    必须直接报错：并发写同名文件会静默丢帧。
    """
    assigned: Dict[Path, str] = {}
    owner: Dict[str, Path] = {}
    for video in videos:
        prefix = video_output_prefix(video, input_dir)
        if prefix in owner:
            raise ValueError(
                f"输出名冲突：{owner[prefix]} 与 {video} 都会写成 {prefix}_frame_*.jpg，"
                "并发写入会互相覆盖；请重命名其中一个，或调整目录结构"
            )
        owner[prefix] = video
        assigned[video] = prefix
    return assigned


def blur_score(frame) -> float:
    """Laplacian 方差，越大越清晰。

    calibrate_blur.py 也调用这个函数算分位数 —— 必须共用同一个实现，
    否则标定出的分布与实际过滤时看到的分数不是一回事，标定就没有意义了。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_frame_blurry(frame, threshold: float) -> bool:
    # blur_score 已转成内置 float，比较结果就是内置 bool
    return blur_score(frame) < threshold


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
    fname = f"{video_name}_frame_{frame_number:06d}.jpg"
    out_path = out_dir / fname
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
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _atomic_write_json(path: Path, payload: dict) -> bool:
    """先写临时文件再替换，避免中途崩溃留下半截 JSON。成功返回 True。"""
    tmp_path = path.parent / (path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, path)
    except OSError:
        logging.getLogger("clean_video").exception("写入 %s 失败", path)
        return False
    return True


def write_images_meta(
    out_images_dir: Path,
    fps_by_video: Dict[str, float],
    target_fps: int,
    results: Sequence[VideoResult],
) -> None:
    """把运行参数和每个视频的原始帧率写到 out_images_dir/_meta.json。

    必须由父进程在收齐所有 future 后调一次：子进程各写各的会互相覆盖。
    已有条目保留（支持增量追加），orig_fps 为 0 的失败视频不写入。

    format 1 → 2 的变化：删掉了 blur_threshold（阈值概念已移到 format_dataset.py），
    删掉 saved（恒等于 sampled）和 skipped_blurry，新增 score_median。
    跨版本增量追加时，旧版本的条目会原样保留（它们的分数没存过，median 无法补算）。
    """
    logger = logging.getLogger("clean_video")
    meta_path = out_images_dir / IMAGES_META_FILENAME

    payload: dict = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            logger.warning("已有的 %s 无法解析，将重新生成", meta_path)

    videos = payload.get("videos")
    if not isinstance(videos, dict):
        videos = {}

    stats = {r.video_name: r for r in results}
    for video_name, orig_fps in fps_by_video.items():
        if orig_fps <= 0:
            continue
        result = stats.get(video_name)
        if result is None:
            continue
        videos[video_name] = {
            "orig_fps": round(float(orig_fps), 6),
            "frame_step": max(1, int(round(orig_fps / float(target_fps)))) if target_fps > 0 else 1,
            "sampled": result.sampled,
            "score_median": round(median_score(result.scores), 4),
            "elapsed": round(result.elapsed, 3),
        }

    payload.update(
        {
            "format": 2,
            "tool": "clean_video.py",
            "target_fps": target_fps,
            "videos": videos,
        }
    )

    if _atomic_write_json(meta_path, payload):
        logger.info("已写入 %s（累计 %d 个视频的原始帧率）", meta_path, len(videos))


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

    try:
        output_names = assign_output_names(videos, input_dir)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    nested = [vid for vid in videos if output_names[vid] != vid.stem]
    if nested:
        logger.info(
            "有 %d 个视频位于子目录，输出名前缀带上了目录：%s",
            len(nested),
            ", ".join(f"{vid.name} -> {output_names[vid]}" for vid in nested[:5]),
        )

    total_sampled = 0
    results: List[VideoResult] = []
    fps_by_video: Dict[str, float] = {}

    start_all = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(
                process_video_worker,
                str(vid),
                str(out_images_dir),
                args.fps,
                output_names[vid],
            ): vid
            for vid in videos
        }

        for fut in concurrent.futures.as_completed(future_map):
            vid = future_map[fut]
            try:
                result = fut.result()
            except Exception:
                logger.exception("Processing failed for %s", vid)
                continue

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

    write_images_meta(out_images_dir, fps_by_video, args.fps, results)
    write_blur_scores(out_images_dir, results)
    return 0


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
