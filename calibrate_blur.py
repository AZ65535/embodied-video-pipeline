#!/usr/bin/env python
"""为 clean_video.py 的模糊阈值标定提供依据。

这是管线之外的辅助工具，不产出数据集。它回答三个问题：

1. 素材的 Laplacian 方差分布长什么样（分位数 + 直方图）
2. 给定的阈值会拦掉多少帧
3. 被拦掉的究竟是真模糊还是误杀 —— 导出复核帧供肉眼判断

**不自动推荐阈值**：Laplacian 方差分布通常不是双峰的，任何自动选阈值的做法
都是在假装有依据。工具只提供事实，线画在哪由人决定 —— 这也是要导出复核帧的原因。

分数由 clean_video.blur_score 计算，与过滤时用的是同一个函数；采样帧号也与
clean_video 一致。两者不一致的话，统计出的分布就对不上实际过滤行为。
"""

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from clean_video import blur_score, list_videos, video_output_prefix

# 默认全量分析。抽样（--sample）只在素材大到全量解码吃不消时才值得用：
# 实测 1080p 长 GOP 视频，跳转读一帧约 59ms，顺序解码一帧约 28ms（且只在候选帧
# 上算分）。跳转能省的总时间有限，却会丢掉大量帧，反而让分位数不准。
DEFAULT_SAMPLE = 0
DEFAULT_THRESHOLD = 100.0
DEFAULT_BAND = 3.0
DEFAULT_EXPORT_MAX = 150

# 分位数取这些点
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)

# 直方图分桶。Laplacian 方差跨几个数量级，等宽分桶会挤成一团，所以用近似对数的边界。
HIST_BUCKETS = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, float("inf"))


@dataclass
class VideoScores:
    """一个视频的采样结果。"""

    path: Path
    name: str                # 与 clean_video.py 的 _meta.json 键一致，便于对照
    orig_fps: float
    frame_step: int
    candidates: int          # 符合采样条件的帧总数（抽样前）
    frame_indices: List[int]
    scores: List[float]
    elapsed: float


def evenly_spaced(items: Sequence, limit: int) -> List:
    """从序列里均匀跨度取 limit 个（含首尾）。

    不取前 N 个：素材开头往往是静止镜头，分布与全片差别很大。复核帧也用它 ——
    按分数排序后再取，选出来的子集仍然覆盖整个分数范围。
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / (limit - 1) if limit > 1 else 0
    return sorted({items[round(i * step)] for i in range(limit)})


def score_video(
    video_path: Path, input_dir: Path, target_fps: int, sample: int
) -> Optional[VideoScores]:
    """采样帧并算出每帧的 Laplacian 方差。

    采样帧号与 clean_video.py 完全一致（`read_idx % frame_step == 0`）。抽样时用
    跳转读取而非全量解码 —— 已实测两者结果逐位相同，且跳转快得多。
    """
    started = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return None

        if orig_fps <= 0 or target_fps <= 0:
            frame_step = 1
        else:
            frame_step = max(1, int(round(orig_fps / float(target_fps))))

        candidates = list(range(0, total_frames, frame_step))
        chosen = evenly_spaced(candidates, sample)

        indices: List[int] = []
        scores: List[float] = []
        if len(chosen) == len(candidates):
            # 本来就要全部，顺序解码比逐个跳转快
            read_idx = 0
            wanted = set(chosen)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if read_idx in wanted:
                    indices.append(read_idx)
                    scores.append(blur_score(frame))
                read_idx += 1
        else:
            for idx in chosen:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                indices.append(idx)
                scores.append(blur_score(frame))
    finally:
        cap.release()

    return VideoScores(
        path=video_path,
        name=video_output_prefix(video_path, input_dir),
        orig_fps=orig_fps,
        frame_step=frame_step,
        candidates=len(candidates),
        frame_indices=indices,
        scores=scores,
        elapsed=time.time() - started,
    )


def score_video_worker(
    video_path_str: str, input_dir_str: str, target_fps: int, sample: int
) -> Optional[VideoScores]:
    """可被进程池调用的包装（top-level）。"""
    return score_video(Path(video_path_str), Path(input_dir_str), target_fps, sample)


def percentiles(
    scores: Sequence[float], points: Sequence[float] = PERCENTILES
) -> List[Tuple[float, float]]:
    """百分位分位数，线性插值。"""
    if not scores:
        return []
    ordered = sorted(scores)
    result = []
    for point in points:
        if point <= 0:
            value = ordered[0]
        elif point >= 100:
            value = ordered[-1]
        else:
            position = (len(ordered) - 1) * point / 100.0
            low, high = math.floor(position), math.ceil(position)
            if low == high:
                value = ordered[low]
            else:
                value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        result.append((point, value))
    return result


def histogram(
    scores: Sequence[float], buckets: Sequence[float] = HIST_BUCKETS
) -> List[Tuple[str, int]]:
    """按固定边界分桶计数，返回 [(区间标签, 数量), ...]。"""
    counts = [0] * (len(buckets) - 1)
    for score in scores:
        for i in range(len(buckets) - 1):
            if buckets[i] <= score < buckets[i + 1]:
                counts[i] += 1
                break
    return [
        (f"{buckets[i]:g}-{buckets[i + 1]:g}", counts[i]) for i in range(len(counts))
    ]


def filter_rate(scores: Sequence[float], threshold: float) -> float:
    """该阈值下会被判为模糊的比例。"""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s < threshold) / len(scores)


def print_distribution(result: VideoScores, logger: logging.Logger) -> None:
    logger.info(
        "--- %s：%d 帧，%.2f fps，步长 %d，采样 %d / %d 帧（%.1fs）---",
        result.name,
        result.candidates * result.frame_step,
        result.orig_fps,
        result.frame_step,
        len(result.scores),
        result.candidates,
        result.elapsed,
    )
    points = percentiles(result.scores)
    logger.info("  " + "  ".join(f"p{int(p)}={v:g}" for p, v in points))

    buckets = histogram(result.scores)
    peak = max((count for _, count in buckets), default=0) or 1
    for label, count in buckets:
        bar = "█" * max(1, round(count / peak * 36)) if count else ""
        logger.info("  %-14s |%s %d", label, bar, count)


def print_threshold_table(
    results: Sequence[VideoScores], thresholds: Sequence[float], logger: logging.Logger
) -> None:
    header = f"{'阈值':>10}" + "".join(f"{r.name:>12}" for r in results) + f"{'合计':>12}"
    logger.info(header)
    for threshold in thresholds:
        row = f"{threshold:>10g}"
        total_below = 0
        total = 0
        for result in results:
            below = sum(1 for s in result.scores if s < threshold)
            total_below += below
            total += len(result.scores)
            rate = below / len(result.scores) * 100 if result.scores else 0.0
            row += f"{rate:>11.1f}%"
        overall = total_below / total * 100 if total else 0.0
        logger.info(row + f"{overall:>11.1f}%")


def export_review_frames(
    result: VideoScores,
    below_dir: Path,
    critical_dir: Path,
    threshold: float,
    band: float,
    export_max: int,
) -> Tuple[int, int, int, int]:
    """把复核帧存出来，返回 (below 写出, critical 写出, below 总数, critical 总数)。

    总数与实际写出数分开返回：截取后屏幕上要能看出「符合条件的有多少、导出了多少」。

    文件名以分数开头（`var0000183.4_1_frame000123.jpg`），按名称排序即按质量排序，
    翻一遍就能看出线该画在哪。

    两类以阈值为界、互不重叠，合起来是一条连续的分数梯度：`below/` 是被判为模糊的
    （看末尾几个，即最接近阈值的，判断有没有误杀）；`critical/` 是刚过线的（看开头
    几个，判断有没有漏网）。两边都按分数排序，所以阈值正好落在两组文件的交界处。

    **分类截取**而非合并截取 —— 否则数量多的一类会把另一类挤掉。

    export_max 上限是必要的：阈值 100 的临界带能覆盖大半个视频，不截取会导出上千个
    文件，根本翻不完。按分数排序后再均匀取，梯度仍然完整。
    """
    high = threshold * band
    pairs = list(zip(result.frame_indices, result.scores))
    below = evenly_spaced(
        sorted((p for p in pairs if p[1] < threshold), key=lambda item: item[1]),
        export_max,
    )
    critical = evenly_spaced(
        sorted((p for p in pairs if threshold <= p[1] <= high), key=lambda item: item[1]),
        export_max,
    )
    total_below = sum(1 for _, score in pairs if score < threshold)
    total_critical = sum(1 for _, score in pairs if threshold <= score <= high)
    if not below and not critical:
        return 0, 0, total_below, total_critical

    below_dir.mkdir(parents=True, exist_ok=True)
    critical_dir.mkdir(parents=True, exist_ok=True)

    counts = {"below": 0, "critical": 0}
    cap = cv2.VideoCapture(str(result.path))
    if not cap.isOpened():
        return 0, 0
    try:
        for kind, group in (("below", below), ("critical", critical)):
            for index, score in group:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                target = below_dir if kind == "below" else critical_dir
                # 定宽是排序正确的前提：文件名按字典序排 = 按分数排。两位小数是为了
                # 让阈值两侧不至于都显示成 100.0（实际 99.96 会被 .1f 显示成 100.0，
                # 看起来像阈值没生效）。分数超过 5 位整数时定宽失效，但 Laplacian
                # 方差不会到那个量级。
                name = f"var{score:09.2f}_{result.name}_frame{index:06d}.jpg"
                if cv2.imwrite(str(target / name), frame):
                    counts[kind] += 1
    finally:
        cap.release()
    return counts["below"], counts["critical"], total_below, total_critical


def write_scores_jsonl(
    path: Path, results: Sequence[VideoScores], thresholds: Sequence[float]
) -> None:
    """每帧一行，供复现或另行分析。"""
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            for index, score in zip(result.frame_indices, result.scores):
                handle.write(
                    json.dumps(
                        {
                            "video": result.name,
                            "source_frame": index,
                            "blur_score": round(score, 4),
                            "below": {
                                f"{t:g}": score < t for t in thresholds
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 clean_video.py 的模糊阈值标定提供依据（分位数、直方图、阈值影响、复核帧）。"
    )
    parser.add_argument("--input", "-i", required=True, help="视频目录，递归扫描 .mp4")
    parser.add_argument(
        "--fps", type=int, default=5, help="采样帧率，应与 clean_video.py 一致（默认 5）"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=(
            "每个视频最多分析多少帧，均匀跨度取；0 = 全量（默认）。"
            "仅当素材大到全量解码吃不消时才设，抽样会让分位数变粗"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help=f"要评估的阈值，可重复传入以对比；默认 [{DEFAULT_THRESHOLD:g}]",
    )
    parser.add_argument("--export-dir", default=None, help="导出复核帧到该目录（不给就不导出）")
    parser.add_argument(
        "--export-band",
        type=float,
        default=DEFAULT_BAND,
        help=f"临界帧上界倍数：导出 [阈值, 阈值*band]（默认 {DEFAULT_BAND:g}）",
    )
    parser.add_argument(
        "--export-max",
        type=int,
        default=DEFAULT_EXPORT_MAX,
        help=(
            "每视频每类最多导出多少帧，按分数均匀取；0 = 不截取"
            f"（默认 {DEFAULT_EXPORT_MAX}）。临界带可能覆盖大半个视频，不截取会翻不完"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="并行进程数（默认 CPU 核数）"
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("calibrate_blur")

    input_dir = Path(args.input)
    thresholds = args.threshold if args.threshold else [DEFAULT_THRESHOLD]

    if args.fps <= 0:
        logger.error("--fps 必须为正整数")
        return 1
    if args.sample < 0:
        logger.error("--sample 不能为负数（0 表示全量）")
        return 1
    if args.export_band <= 1:
        logger.error("--export-band 必须大于 1（否则 [阈值, 阈值*band] 退化成一个点）")
        return 1
    if args.export_max < 0:
        logger.error("--export-max 不能为负数（0 表示不截取）")
        return 1
    for threshold in thresholds:
        if threshold <= 0:
            logger.error("--threshold 必须为正数，收到 %g", threshold)
            return 1

    try:
        videos = list(list_videos(input_dir))
    except Exception:
        logger.exception("扫描视频目录失败：%s", input_dir)
        return 1
    if not videos:
        logger.error("在 %s 中没有找到 mp4 视频", input_dir)
        return 1

    logger.info(
        "分析 %d 个视频，采样帧率 %d，每视频最多 %s 帧",
        len(videos),
        args.fps,
        "全量" if args.sample <= 0 else args.sample,
    )

    started = time.time()
    results: List[VideoScores] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                score_video_worker, str(video), str(input_dir), args.fps, args.sample
            ): video
            for video in videos
        }
        for future in as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.exception("分析失败：%s", video)
                continue
            if result is None:
                logger.warning("无法读取，跳过：%s", video)
                continue
            results.append(result)

    if not results:
        logger.error("没有任何视频被成功分析")
        return 1

    results.sort(key=lambda r: r.name)

    logger.info("\n=== 方差分布 ===")
    for result in results:
        print_distribution(result, logger)

    logger.info("\n=== 阈值影响（被判为模糊、丢弃的比例）===")
    print_threshold_table(results, thresholds, logger)

    logger.info("\n=== 各视频一致性 ===")
    medians = [(r.name, percentiles(r.scores)[PERCENTILES.index(50)][1]) for r in results]
    for name, median in medians:
        logger.info("  %-14s 中位数 %g", name, median)
    if len(medians) > 1:
        values = [m for _, m in medians]
        if min(values) > 0 and max(values) / min(values) >= 2:
            logger.warning(
                "各视频的方差中位数相差 %.1f 倍。同一个绝对阈值对它们的严格程度并不一致，"
                "统一阈值意味着各视频的过滤标准其实不同 —— 可考虑按视频分别定阈值",
                max(values) / min(values),
            )

    if args.export_dir:
        export_dir = Path(args.export_dir)
        below_dir = export_dir / "below"
        critical_dir = export_dir / "critical"
        logger.info("\n=== 导出复核帧 -> %s（按阈值 %g）===", export_dir, thresholds[0])
        written_below = written_critical = total_below = total_critical = 0
        for result in results:
            below, critical, all_below, all_critical = export_review_frames(
                result,
                below_dir,
                critical_dir,
                thresholds[0],
                args.export_band,
                args.export_max,
            )
            written_below += below
            written_critical += critical
            total_below += all_below
            total_critical += all_critical
            logger.info(
                "  %-14s below=%d/%d  critical=%d/%d",
                result.name,
                below,
                all_below,
                critical,
                all_critical,
            )
        try:
            write_scores_jsonl(export_dir / "scores.jsonl", results, thresholds)
        except OSError:
            logger.exception("写入 scores.jsonl 失败")
        logger.info(
            "  below/ 是被判为模糊的（符合 %d 帧，导出 %d）；"
            "critical/ 是刚过线的（符合 %d 帧，导出 %d）",
            total_below,
            written_below,
            total_critical,
            written_critical,
        )
        logger.info(
            "  文件名以分数开头（var0000183.4_...），按名称排序即按质量排序。"
            "阈值就在 below/ 末尾与 critical/ 开头的交界处"
        )

    logger.info("\n=== 完成（耗时 %.1fs）===", time.time() - started)
    logger.info(
        "本工具不改任何阈值。定好之后用 clean_video.py --blur_threshold <值> 重新抽帧"
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
