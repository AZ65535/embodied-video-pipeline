#!/usr/bin/env python
"""为 format_dataset.py 的模糊阈值标定提供依据。

这是管线之外的辅助工具，不产出数据集。它回答三个问题：

1. 素材的 Laplacian 方差分布长什么样（分位数 + 直方图）
2. 给定的阈值会把多少帧标为模糊
3. 越线的究竟是真模糊还是误杀 —— 导出复核帧供肉眼判断

**不自动推荐阈值**：Laplacian 方差分布通常不是双峰的，任何自动选阈值的做法
都是在假装有依据。工具只提供事实，线画在哪由人决定 —— 这也是要导出复核帧的原因。

输入是 clean_video.py 的产物目录（`output/images`），**不重新解码视频**：每帧的分数
第一步已经算好并存进 `_scores.jsonl` 了，这里只是把它读出来重新统计。因此本模块
不需要 cv2，与 format_dataset.py 一样是纯标准库。

早先的版本会自己解码视频来算分，那时「采样规则必须与 clean_video 一致」是靠两处
代码写同一套逻辑维持的约定。现在读的是同一份数据，一致性是事实而不是约定。
"""

import argparse
import json
import logging
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pipeline_common import (
    BLUR_SCORES_FILENAME,
    DEFAULT_BLUR_THRESHOLD,
    IMAGES_META_FILENAME,
    frame_filename,
    is_blurry,
    read_blur_scores,
    setup_logging,
    write_jsonl,
)

DEFAULT_THRESHOLD = DEFAULT_BLUR_THRESHOLD
DEFAULT_BAND = 3.0
DEFAULT_EXPORT_MAX = 150

# 分位数取这些点
PERCENTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)

# 直方图分桶。Laplacian 方差跨几个数量级，等宽分桶会挤成一团，所以用近似对数的边界。
HIST_BUCKETS = (0.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, float("inf"))


@dataclass
class VideoScores:
    """一个视频（第一步的输出名前缀）的分数。

    字段比解码版少：分数是现成的，没有「解码耗时」「候选帧数」这类过程量。
    """

    name: str                                    # 与 clean_video.py 的 _meta.json 键一致
    frame_indices: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    # 下面两项只为在报告里显示，来自 _meta.json；缺失时为 None
    orig_fps: Optional[float] = None
    frame_step: Optional[int] = None


def evenly_spaced(items: Sequence, limit: int) -> List:
    """从序列里均匀跨度取 limit 个（含首尾）。

    不取前 N 个：素材开头往往是静止镜头，分布与全片差别很大。复核帧也用它 ——
    按分数排序后再取，选出来的子集仍然覆盖整个分数范围。
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    step = (len(items) - 1) / (limit - 1) if limit > 1 else 0
    return sorted({items[round(i * step)] for i in range(limit)})


def load_scores(images_dir: Path) -> List[VideoScores]:
    """读 clean_video.py 写的 _scores.jsonl，按视频分组后按帧号排序。"""
    logger = logging.getLogger("calibrate_blur")
    scores_path = images_dir / BLUR_SCORES_FILENAME
    if not scores_path.is_file():
        raise FileNotFoundError(
            f"找不到 {scores_path}。本工具分析的是第一步的产物，"
            "请先跑 clean_video.py --input <视频目录> --output <输出目录>"
        )

    grouped = read_blur_scores(scores_path, logger)
    if not grouped:
        raise ValueError(f"{scores_path} 里没有任何可用的分数")

    # _meta.json 只为在报告里显示原始帧率，读不到不影响分析
    meta_by_video = {}
    meta_path = images_dir / IMAGES_META_FILENAME
    if meta_path.is_file():
        try:
            videos = json.loads(meta_path.read_text(encoding="utf-8")).get("videos")
            if isinstance(videos, dict):
                meta_by_video = videos
        except (OSError, json.JSONDecodeError, AttributeError):
            logger.warning("%s 无法解析，报告里不显示原始帧率", meta_path)

    results: List[VideoScores] = []
    for name in sorted(grouped):
        frame_scores = sorted(grouped[name].items())
        meta = meta_by_video.get(name) if isinstance(meta_by_video.get(name), dict) else {}
        results.append(
            VideoScores(
                name=name,
                frame_indices=[frame for frame, _ in frame_scores],
                scores=[score for _, score in frame_scores],
                orig_fps=meta.get("orig_fps"),
                frame_step=meta.get("frame_step"),
            )
        )

    logger.info(
        "已读取 %s：%d 个视频、%d 帧（第一步算好的分数，不重新解码）",
        scores_path,
        len(results),
        sum(len(r.scores) for r in results),
    )
    return results


def percentiles(
    scores: Sequence[float], points: Sequence[float] = PERCENTILES
) -> List[Tuple[float, float]]:
    """百分位（线性插值）。空列表返回空。"""
    if not scores:
        return []
    ordered = sorted(scores)
    last = len(ordered) - 1
    result: List[Tuple[float, float]] = []
    for point in points:
        position = point / 100.0 * last
        low = int(position)
        high = min(low + 1, last)
        weight = position - low
        value = ordered[low] * (1 - weight) + ordered[high] * weight
        result.append((point, value))
    return result


def histogram(
    scores: Sequence[float], buckets: Sequence[float] = HIST_BUCKETS
) -> List[Tuple[str, int]]:
    """按 buckets 分桶计数。桶边界归上一桶（左闭右开）。"""
    counts = [0] * (len(buckets) - 1)
    for score in scores:
        for index in range(len(buckets) - 1):
            if buckets[index] <= score < buckets[index + 1]:
                counts[index] += 1
                break
        else:
            counts[-1] += 1     # 理论上到不了：最后一桶上界是 inf
    labels = [
        f"{buckets[i]:g}-{buckets[i + 1]:g}" if buckets[i + 1] != float("inf")
        else f"{buckets[i]:g}+"
        for i in range(len(buckets) - 1)
    ]
    return list(zip(labels, counts))


def filter_rate(scores: Sequence[float], threshold: float) -> float:
    """低于阈值的比例（严格小于：等于阈值算清晰）。口径见 pipeline_common.is_blurry。"""
    if not scores:
        return 0.0
    return sum(1 for s in scores if is_blurry(s, threshold)) / len(scores)


def print_distribution(result: VideoScores, logger: logging.Logger) -> None:
    if result.orig_fps and result.frame_step:
        origin = f"{result.orig_fps:g} fps，步长 {result.frame_step}"
    else:
        origin = "原始帧率未知"
    logger.info("--- %s：%d 帧（%s）---", result.name, len(result.scores), origin)

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
    all_scores = [s for r in results for s in r.scores]
    for threshold in thresholds:
        row = f"{threshold:>10g}"
        for result in results:
            row += f"{filter_rate(result.scores, threshold) * 100:>11.1f}%"
        logger.info(row + f"{filter_rate(all_scores, threshold) * 100:>11.1f}%")


def export_review_frames(
    result: VideoScores,
    images_dir: Path,
    below_dir: Path,
    critical_dir: Path,
    threshold: float,
    band: float,
    export_max: int,
) -> Tuple[int, int, int, int]:
    """把复核帧拷出来，返回 (below 写出, critical 写出, below 总数, critical 总数)。

    总数与实际写出数分开返回：截取后屏幕上要能看出「符合条件的有多少、导出了多少」。
    帧是从第一步的产物目录**拷贝**的，不再解码视频。

    文件名以分数开头（`var000183.42_1_frame000123.jpg`），按名称排序即按质量排序，
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

    logger = logging.getLogger("calibrate_blur")
    counts = {"below": 0, "critical": 0}
    for kind, group in (("below", below), ("critical", critical)):
        target = below_dir if kind == "below" else critical_dir
        for index, score in group:
            source = images_dir / frame_filename(result.name, index)
            if not source.is_file():
                logger.warning("帧文件不存在，跳过：%s", source)
                continue
            # 定宽是排序正确的前提：文件名按字典序排 = 按分数排。两位小数是为了
            # 让阈值两侧不至于都显示成 100.0（实际 99.96 会被 .1f 显示成 100.0，
            # 看起来像阈值没生效）。分数超过 5 位整数时定宽失效，但 Laplacian
            # 方差不会到那个量级。
            name = f"var{score:09.2f}_{result.name}_frame{index:06d}.jpg"
            try:
                shutil.copy2(source, target / name)
            except OSError:
                logger.exception("拷贝失败：%s", source)
                continue
            counts[kind] += 1
    return counts["below"], counts["critical"], total_below, total_critical


def write_scores_jsonl(
    path: Path, results: Sequence[VideoScores], thresholds: Sequence[float]
) -> None:
    """每帧一行，供复现或另行分析。"""
    rows = (
        {
            "video": result.name,
            "source_frame": index,
            "blur_score": round(score, 4),
            "below": {f"{t:g}": is_blurry(score, t) for t in thresholds},
        }
        for result in results
        for index, score in zip(result.frame_indices, result.scores)
    )
    write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 format_dataset.py 的模糊阈值标定提供依据"
        "（分位数、直方图、阈值影响、复核帧）。"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="output/images",
        help="clean_video.py 的产物目录，即含 _scores.jsonl 的图片目录（默认 output/images）",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = logging.getLogger("calibrate_blur")

    images_dir = Path(args.input)
    thresholds = args.threshold if args.threshold else [DEFAULT_THRESHOLD]

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

    started = time.time()

    try:
        results = load_scores(images_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except (OSError, ValueError) as exc:
        logger.error("读取失败：%s", exc)
        return 1

    logger.info("\n=== 方差分布 ===")
    for result in results:
        print_distribution(result, logger)

    logger.info("\n=== 阈值影响（被判为模糊、会被标掉的比例）===")
    print_threshold_table(results, thresholds, logger)

    logger.info("\n=== 各视频一致性 ===")
    medians = [(r.name, statistics.median(r.scores)) for r in results if r.scores]
    for name, median in medians:
        logger.info("  %-14s 中位数 %g", name, median)
    if len(medians) > 1:
        values = [m for _, m in medians]
        if min(values) > 0 and max(values) / min(values) >= 2:
            logger.warning(
                "各视频的方差中位数相差 %.1f 倍。同一个绝对阈值对它们的严格程度并不一致，"
                "统一阈值意味着各视频的标注标准其实不同 —— 可考虑按视频分别定阈值",
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
                images_dir,
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
            "  文件名以分数开头（var000183.42_...），按名称排序即按质量排序。"
            "阈值就在 below/ 末尾与 critical/ 开头的交界处"
        )

    logger.info("\n=== 完成（耗时 %.1fs）===", time.time() - started)
    logger.info(
        "本工具不改任何阈值。定好之后用 format_dataset.py --blur-threshold <值> 重新生成清单"
        "（帧不会因此被删，只是 frames.jsonl 里的 is_blurry 列换一种填法）"
    )
    return 0


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
