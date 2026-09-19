import argparse
import concurrent.futures
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List
import time

import cv2
from tqdm import tqdm


# 处理参数与原始帧率的落盘文件名，写在图片目录里，供 format_dataset.py 读取。
IMAGES_META_FILENAME = "_meta.json"


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


def video_output_prefix(video_path: Path, input_dir: Path) -> str:
    """算出该视频的输出名前缀（图片文件名和 _meta.json 的键都用它）。

    规则：相对 input_dir 的路径去掉后缀，再把路径分隔符换成下划线。
    平铺目录下结果就等于原来的 stem（行为不变）；子目录下会带上目录名，
    这样 videos/a/1.mp4 与 videos/b/1.mp4 不会写成同一批文件名互相覆盖。
    """
    try:
        relative = video_path.relative_to(input_dir)
    except ValueError:
        return video_path.stem  # 不在输入目录下（理论上不会发生）
    return "_".join(relative.with_suffix("").parts)


def assign_output_names(videos: Iterable[Path], input_dir: Path) -> Dict[Path, str]:
    """给每个视频分配输出前缀；仍有重名时抛 ValueError。

    带上目录前缀之后仍可能撞车，例如 videos/a_1.mp4 与 videos/a/1.mp4 都会得到 "a_1"。
    这种情况必须直接报错：两个进程并发写同名文件会静默丢帧，绝不能放过去。
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


def is_frame_blurry(frame, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()
    # lap.var() 是 numpy 标量，比较结果是 np.bool_ 而非内置 bool；
    # 显式转换以符合类型标注，也避免调用方 is True 之类的判断意外失败。
    return bool(score < threshold)


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
    output_name: Optional[str] = None,
) -> Tuple[int, int, int, float, str, float]:
    """抽帧并过滤模糊帧，保存清晰帧。

    返回 (sampled, saved, skipped_blurry, elapsed, video_name, orig_fps)；
    视频无法打开等失败情况下计数为 0、orig_fps 为 0.0。

    output_name 是输出文件名前缀（同时用作 _meta.json 的键），缺省用去后缀的文件名。
    由调用方经 assign_output_names 统一分配，避免不同子目录的同名视频互相覆盖。
    """
    logger = logging.getLogger(__name__)
    # 提前定义：异常路径下的 except 块需要返回它们
    video_name = output_name or video_path.stem
    saved = 0
    skipped = 0
    sampled = 0
    orig_fps = 0.0
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

        return sampled, saved, skipped, elapsed, video_name, orig_fps

    except Exception:
        logger.exception("Failed to process video: %s", video_path)
        elapsed = time.time() - start_ts
        return sampled, saved, skipped, elapsed, video_name, orig_fps

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


def write_images_meta(
    out_images_dir: Path,
    fps_by_video: Dict[str, float],
    target_fps: int,
    blur_threshold: float,
    results: List[Tuple[str, int, int, int, float]],
) -> None:
    """把本次的运行参数和每个视频的原始帧率写到 out_images_dir/_meta.json。

    必须由父进程在所有 future 收集完毕后调用一次：子进程各写各的会互相覆盖。
    已存在的条目会保留（支持增量追加到同一目录），orig_fps 为 0 的失败视频不写入。
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

    stats = {
        name: (sampled, saved, skipped, elapsed)
        for name, sampled, saved, skipped, elapsed in results
    }
    for video_name, orig_fps in fps_by_video.items():
        if orig_fps <= 0:
            continue
        sampled, saved, skipped, elapsed = stats.get(video_name, (0, 0, 0, 0.0))
        videos[video_name] = {
            "orig_fps": round(float(orig_fps), 6),
            "frame_step": max(1, int(round(orig_fps / float(target_fps)))) if target_fps > 0 else 1,
            "sampled": sampled,
            "saved": saved,
            "skipped_blurry": skipped,
            "elapsed": round(elapsed, 3),
        }

    payload.update(
        {
            "format": 1,
            "tool": "clean_video.py",
            "target_fps": target_fps,
            "blur_threshold": blur_threshold,
            "videos": videos,
        }
    )

    # 先写临时文件再替换，避免中途崩溃留下半截 JSON
    tmp_path = meta_path.parent / (meta_path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, meta_path)
    except OSError:
        logger.exception("写入 %s 失败", meta_path)
        return

    logger.info("已写入 %s（累计 %d 个视频的原始帧率）", meta_path, len(videos))


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
    total_saved = 0
    total_skipped = 0
    results: List[Tuple[str, int, int, int, float]] = []
    fps_by_video: Dict[str, float] = {}

    start_all = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(
                process_video_worker,
                str(vid),
                str(out_images_dir),
                args.fps,
                args.blur_threshold,
                output_names[vid],
            ): vid
            for vid in videos
        }

        for fut in concurrent.futures.as_completed(future_map):
            vid = future_map[fut]
            try:
                sampled, saved, skipped, elapsed, video_name, orig_fps = fut.result()
            except Exception:
                logger.exception("Processing failed for %s", vid)
                continue

            if orig_fps > 0:
                fps_by_video[video_name] = orig_fps
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

    write_images_meta(out_images_dir, fps_by_video, args.fps, args.blur_threshold, results)
    return 0


def process_video_worker(
    video_path_str: str,
    out_images_dir_str: str,
    target_fps: int,
    blur_threshold: float,
    output_name: str,
):
    """可被进程池安全调用的包装函数（top-level）。"""
    video_path = Path(video_path_str)
    out_images_dir = Path(out_images_dir_str)
    return process_video(
        video_path,
        out_images_dir,
        target_fps=target_fps,
        blur_threshold=blur_threshold,
        output_name=output_name,
    )


if __name__ == "__main__":
    setup_logging()
    sys.exit(main())
