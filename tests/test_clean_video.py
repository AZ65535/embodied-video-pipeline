"""clean_video.py 的测试。

需要 cv2，所以单独成文件 —— format_dataset 是纯标准库的，
不希望它的测试因为这里的导入失败而一起挂掉。

视频用 cv2.VideoWriter 现场合成，不依赖仓库里的 videos/。
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

import clean_video

WIDTH, HEIGHT = 64, 48
BLOCK = 8


def sharp_frame() -> np.ndarray:
    """高对比度方波条纹。块取得大，是为了经得起 mp4 压缩后依然保持高方差。"""
    frame = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = 235 if (y // BLOCK) % 2 == 0 else 20
    return frame


def flat_frame() -> np.ndarray:
    """纯色块，Laplacian 方差接近 0，必然被判为模糊。"""
    return np.full((HEIGHT, WIDTH, 3), 128, np.uint8)


@pytest.fixture
def make_video(tmp_path: Path):
    """把一串帧写成 mp4，返回文件路径。"""

    def _make(name: str, frames: list, fps: int = 30) -> Path:
        path = tmp_path / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
        )
        if not writer.isOpened():
            pytest.skip("当前 OpenCV 写不出 mp4（缺少编码器）")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        assert path.is_file() and path.stat().st_size > 0, "合成的视频是空文件"
        return path

    return _make


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    path = tmp_path / "images"
    path.mkdir()
    return path


# --------------------------------------------------------------------------
# 模糊判定
# --------------------------------------------------------------------------


def test_flat_frame_is_blurry():
    assert clean_video.is_frame_blurry(flat_frame(), 100.0) is True


def test_sharp_frame_is_not_blurry():
    assert clean_video.is_frame_blurry(sharp_frame(), 100.0) is False


def test_threshold_above_sharp_score_flags_it():
    """阈值高到离谱时，再锐的帧也该被判模糊。"""
    assert clean_video.is_frame_blurry(sharp_frame(), 1e9) is True


# --------------------------------------------------------------------------
# 视频扫描
# --------------------------------------------------------------------------


def test_list_videos_matches_each_file_once(tmp_path):
    """回归：历史上 "*.mp4" 与 "*.MP4" 两轮 rglob 会把同一个文件匹配两次、重复处理。"""
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "c.avi").write_bytes(b"x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "d.mp4").write_bytes(b"x")

    assert sorted(p.name for p in clean_video.list_videos(tmp_path)) == ["a.mp4", "b.MP4", "d.mp4"]


def test_list_videos_rejects_missing_directory():
    with pytest.raises(FileNotFoundError):
        list(clean_video.list_videos(Path("no") / "such" / "dir"))


# --------------------------------------------------------------------------
# 抽帧主流程
# --------------------------------------------------------------------------


def test_process_video_keeps_every_sharp_frame(make_video, images_dir):
    video = make_video("sharp.mp4", [sharp_frame() for _ in range(30)])
    sampled, saved, skipped, elapsed, name, orig_fps = clean_video.process_video(
        video, images_dir, target_fps=5, blur_threshold=100.0
    )
    assert sampled > 0
    assert skipped == 0
    assert saved == sampled
    assert name == "sharp"
    assert orig_fps > 0
    assert len(list(images_dir.glob("*.jpg"))) == saved


def test_process_video_drops_every_blurry_frame(make_video, images_dir):
    video = make_video("flat.mp4", [flat_frame() for _ in range(30)])
    sampled, saved, skipped, elapsed, name, orig_fps = clean_video.process_video(
        video, images_dir, target_fps=5, blur_threshold=100.0
    )
    assert sampled > 0
    assert saved == 0
    assert skipped == sampled
    assert not list(images_dir.glob("*.jpg"))


def test_process_video_names_files_by_source_frame_index(make_video, images_dir):
    """文件名里的帧号必须是原始帧号，format_dataset 靠它还原时序空洞。"""
    video = make_video("clip.mp4", [sharp_frame() for _ in range(30)])
    clean_video.process_video(video, images_dir, target_fps=5, blur_threshold=100.0)

    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    assert names, "应当至少抽出一帧"
    # 30fps 采样到 5fps，步长 6：帧号必须是 0, 6, 12, ...
    for index, filename in enumerate(names):
        assert filename == f"clip_frame_{index * 6:06d}.jpg"


def test_process_video_reports_invariant_on_mixed_content(make_video, images_dir):
    frames = [sharp_frame(), flat_frame()] * 15
    video = make_video("mixed.mp4", frames)
    sampled, saved, skipped, elapsed, name, orig_fps = clean_video.process_video(
        video, images_dir, target_fps=5, blur_threshold=100.0
    )
    assert saved + skipped == sampled
    assert len(list(images_dir.glob("*.jpg"))) == saved


def test_process_video_survives_unopenable_file(tmp_path, images_dir):
    """cv2 打开这个文件时，ffmpeg 会往 stderr 打一句 "moov atom not found" —— 属正常噪音。"""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"definitely not a video")

    sampled, saved, skipped, elapsed, name, orig_fps = clean_video.process_video(
        broken, images_dir, target_fps=5, blur_threshold=100.0
    )
    assert (sampled, saved, skipped) == (0, 0, 0)
    assert name == "broken"
    assert orig_fps == 0.0


def test_process_video_creates_output_directory(make_video, tmp_path):
    video = make_video("clip.mp4", [sharp_frame() for _ in range(12)])
    nested = tmp_path / "a" / "b" / "images"          # 事先不存在
    clean_video.process_video(video, nested, target_fps=5, blur_threshold=100.0)
    assert nested.is_dir()


# --------------------------------------------------------------------------
# _meta.json 的写出
# --------------------------------------------------------------------------


def read_meta(images_dir: Path) -> dict:
    return json.loads(
        (images_dir / clean_video.IMAGES_META_FILENAME).read_text(encoding="utf-8")
    )


def test_write_images_meta_records_rates_and_parameters(images_dir):
    clean_video.write_images_meta(
        images_dir, {"1": 30.0}, 5, 100.0, [("1", 1000, 818, 182, 55.7)]
    )
    payload = read_meta(images_dir)
    assert payload["target_fps"] == 5
    assert payload["blur_threshold"] == 100.0

    entry = payload["videos"]["1"]
    assert entry["orig_fps"] == 30.0
    assert entry["frame_step"] == 6         # 30 / 5
    assert entry["sampled"] == 1000
    assert entry["saved"] == 818
    assert entry["skipped_blurry"] == 182


def test_write_images_meta_merges_instead_of_clobbering(images_dir):
    """分批跑同一个输出目录时，先前的记录不能被抹掉。"""
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, 100.0, [])
    clean_video.write_images_meta(images_dir, {"2": 25.0}, 5, 100.0, [])

    assert set(read_meta(images_dir)["videos"]) == {"1", "2"}


def test_write_images_meta_skips_videos_that_failed(images_dir):
    """失败视频的 orig_fps 是 0，不能写进去污染既有记录。"""
    clean_video.write_images_meta(images_dir, {"1": 30.0, "broken": 0.0}, 5, 100.0, [])
    assert set(read_meta(images_dir)["videos"]) == {"1"}


def test_write_images_meta_recovers_from_corrupt_file(images_dir):
    (images_dir / clean_video.IMAGES_META_FILENAME).write_text("{ not json at all", encoding="utf-8")
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, 100.0, [])
    assert read_meta(images_dir)["videos"]["1"]["orig_fps"] == 30.0


def test_write_images_meta_leaves_no_temp_file(images_dir):
    clean_video.write_images_meta(images_dir, {"1": 30.0}, 5, 100.0, [])
    assert list(images_dir.glob("*.tmp")) == []


# --------------------------------------------------------------------------
# 两个模块的接口契约
# --------------------------------------------------------------------------


def test_meta_written_by_clean_video_is_accepted_by_format_dataset(
    make_images, run_format, tmp_path, caplog
):
    """clean_video 写、format_dataset 读 —— 正常路径不应产生任何告警。"""
    images = make_images({"1": [0, 6, 12]})
    clean_video.write_images_meta(images, {"1": 30.0}, 5, 100.0, [("1", 100, 3, 0, 1.0)])

    out = tmp_path / "ds"
    with caplog.at_level(logging.INFO):
        rc = run_format("--input", images, "--output", out)
    assert rc == 0
    assert "已读取" in caplog.text

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"正常路径不该有告警：{warnings}"
