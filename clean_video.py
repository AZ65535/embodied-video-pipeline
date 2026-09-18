import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Iterable, Tuple, List
import time

import cv2
from tqdm import tqdm
import concurrent.futures


def list_videos(input_dir: Path) -> Iterable[Path]:
    """返回输入目录下的所有 mp4 视频路径（递归，后缀大小写不敏感）。

    单次遍历后按后缀过滤：Windows 上 glob 匹配大小写不敏感，
    原先 "*.mp4" / "*.MP4" 两轮 rglob 会把同一文件匹配两次、重复处理。
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            yield p


def is_frame_blurry(frame, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()
    return score < threshold


def save_frame(frame, out_dir: Path, video_name: str, frame_number: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{video_name}_frame_{frame_number:06d}.jpg"
    out_path = out_dir / fname
    # 使用 JPEG 保存，质量默认
    cv2.imwrite(str(out_path), frame)
    return out_path


def process_video(
    video_path: Path,
    out_images_dir: Path,
    target_fps: int = 5,
    blur_threshold: float = 100.0,
) -> Tuple[int, int, int, float, str]:
    """抽帧并过滤模糊帧，保存清晰帧。

    返回 (sampled, saved, skipped_blurry, elapsed, video_name)；
    视频无法打开等失败情况下计数为 0。
    """
    logger = logging.getLogger(__name__)
    # 提前定义：异常路径下的 except 块需要返回它
    video_name = video_path.stem
    saved = 0
    skipped = 0
    sampled = 0
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
                        if is_frame_blurry(frame, blur_threshold):
                            skipped += 1
                        else:
                            save_frame(frame, out_images_dir, video_name, read_idx)
                            saved += 1
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
            "Finished %s: sampled=%d saved=%d skipped_blurry=%d elapsed=%.2fs",
            video_path.name,
            sampled,
            saved,
            skipped,
            elapsed,
        )

        return sampled, saved, skipped, elapsed, video_name

    except Exception:
        logger.exception("Failed to process video: %s", video_path)
        elapsed = time.time() - start_ts
        return sampled, saved, skipped, elapsed, video_name

    finally:
        if cap is not None:
            cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean videos by extracting non-blurry frames.")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing MP4 videos")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output base directory (images will be saved to OUTPUT/images/)",
    )
    parser.add_argument("--fps", type=int, default=5, help="Target frames per second to sample")
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold below which frames are considered blurry",
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


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

    total_sampled = 0
    total_saved = 0
    total_skipped = 0
    results: List[Tuple[str, int, int, int, float]] = []

    start_all = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(
                process_video_worker, str(vid), str(out_images_dir), args.fps, args.blur_threshold
            ): vid
            for vid in videos
        }

        for fut in concurrent.futures.as_completed(future_map):
            vid = future_map[fut]
            try:
                sampled, saved, skipped, elapsed, video_name = fut.result()
            except Exception:
                logger.exception("Processing failed for %s", vid)
                continue

            results.append((video_name, sampled, saved, skipped, elapsed))
            total_sampled += sampled
            total_saved += saved
            total_skipped += skipped
            logger.info(
                "Completed %s: sampled=%d saved=%d skipped=%d elapsed=%.2fs",
                video_name,
                sampled,
                saved,
                skipped,
                elapsed,
            )

    total_elapsed = time.time() - start_all

    # 打印 Benchmark 汇总表
    logger.info("\n=== Benchmark Results ===")
    logger.info("%-40s %8s %8s %9s %10s", "video", "sampled", "saved", "filter%", "time(s)")
    for video_name, sampled, saved, skipped, elapsed in results:
        filter_rate = (skipped / sampled * 100.0) if sampled > 0 else 0.0
        logger.info("%-40s %8d %8d %8.2f%% %10.2f", video_name, sampled, saved, filter_rate, elapsed)

    logger.info(
        "Totals: sampled=%d saved=%d skipped=%d total_time=%.2fs",
        total_sampled,
        total_saved,
        total_skipped,
        total_elapsed,
    )
    return 0


def process_video_worker(video_path_str: str, out_images_dir_str: str, target_fps: int, blur_threshold: float):
    """可被进程池安全调用的包装函数（top-level）。"""
    video_path = Path(video_path_str)
    out_images_dir = Path(out_images_dir_str)
    return process_video(video_path, out_images_dir, target_fps=target_fps, blur_threshold=blur_threshold)


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
